#!/usr/bin/env python3
"""htop-gpu — terminal dashboard for NVIDIA GPUs, CPU/memory, and processes."""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psutil

# Optional fast path: NVML library bindings. Falls back to nvidia-smi
# subprocess if import or initialization fails (older drivers, missing libs,
# stripped-down systems, etc.).
try:
    import pynvml as _nvml
    _HAS_NVML: bool = True
except Exception:  # pragma: no cover — only triggers when the wrapper is missing
    _nvml = None  # type: ignore[assignment]
    _HAS_NVML = False
_nvml_inited: bool = False
_nvml_handles: dict = {}


def _nvml_init() -> bool:
    """Lazily init NVML once per process. Returns True if NVML is usable."""
    global _nvml_inited, _HAS_NVML
    if not _HAS_NVML:
        return False
    if _nvml_inited:
        return True
    try:
        _nvml.nvmlInit()
        _nvml_inited = True
        return True
    except Exception:
        _HAS_NVML = False
        return False


def _nvml_handle(idx: int):
    if idx not in _nvml_handles:
        _nvml_handles[idx] = _nvml.nvmlDeviceGetHandleByIndex(idx)
    return _nvml_handles[idx]


def _decode_str(v) -> str:
    """NVML returns either str or bytes depending on bindings version."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


# ── Terminal helpers ─────────────────────────────────────────────────────────

def _term_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


_COLOR_ON = _use_color()


class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"


def c(code: str, text: str) -> str:
    """Apply color only when supported."""
    if _COLOR_ON:
        return f"{code}{text}{C.RESET}"
    return text


def cb(text: str) -> str:
    """Bold."""
    return c(C.BOLD, text)


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class GpuInfo:
    index: int
    uuid: str
    name: str
    mem_total: int      # MiB
    mem_used: int       # MiB
    mem_free: int       # MiB
    gpu_util: int       # %
    mem_util: int       # %
    temp: int           # °C
    power_draw: float   # W
    power_limit: float  # W
    fan_speed: int      # %


@dataclass
class ProcessInfo:
    pid: int
    gpu_index: int
    gpu_mem: int        # MiB
    user: str = ""
    cmd: str = ""
    cmd_short: str = ""
    cwd: str = ""
    elapsed: str = ""
    started: str = ""
    conda_env: str = ""
    venv: str = ""
    parent_info: str = ""
    container_name: str = ""
    container_image: str = ""
    cpu_percent: float = 0.0   # may exceed 100 on multi-core
    rss_bytes: int = 0          # resident set size


@dataclass
class SystemInfo:
    cpu_per_core: list[float] = field(default_factory=list)
    cpu_total: float = 0.0
    load_avg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mem_total: int = 0
    mem_used: int = 0
    mem_available: int = 0
    mem_percent: float = 0.0
    swap_total: int = 0
    swap_used: int = 0
    swap_percent: float = 0.0
    uptime_seconds: float = 0.0
    cpu_model: str = ""
    cpu_count_logical: int = 0
    cpu_count_physical: int = 0


# ── UI state + click regions (htop-style interactive watch mode) ─────────────

# Sort keys → (attribute, label displayed in column header)
SORT_KEYS = [
    ("gpu_index", "GPU"),
    ("pid", "PID"),
    ("user", "USER"),
    ("gpu_mem", "GPU MEM"),
    ("cpu_percent", "%CPU"),
    ("rss_bytes", "RES"),
    ("elapsed_sec", "ELAPSED"),
    ("cmd_short", "COMMAND"),
]


@dataclass
class UIState:
    filter_gpu: int | None = None   # None = all GPUs
    sort_key: str = "gpu_index"
    sort_desc: bool = False
    long: bool = False
    mode: str = "gpu"               # "gpu" | "cpu" | "mem" — what process list to show
    selected_pid: int | None = None # process selected via click/arrows
    kill_prompt: str | None = None  # transient status text shown in title line
    focus_procs: bool = False       # when True, hide cpu/mem/gpus and fill with procs


@dataclass
class ClickRegion:
    row_min: int        # 1-indexed terminal row, inclusive
    row_max: int        # inclusive
    col_min: int        # 1-indexed terminal col, inclusive
    col_max: int        # inclusive
    action: str         # e.g. "gpu:0", "sort:cpu_percent"


def _parse_etime(s: str) -> int:
    """Parse ps etime format: [DD-]HH:MM:SS or MM:SS. Returns seconds."""
    if not s:
        return 0
    try:
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 3:
            h, m, sec = parts
        elif len(parts) == 2:
            h, m, sec = 0, parts[0], parts[1]
        else:
            h, m, sec = 0, 0, parts[0] if parts else 0
        return days * 86400 + h * 3600 + m * 60 + sec
    except (ValueError, IndexError):
        return 0


def sort_procs(procs: list[ProcessInfo], key: str, desc: bool) -> list[ProcessInfo]:
    """Sort processes by the given key. When sorting by GPU index, a secondary
    sort by GPU memory (largest first within each GPU) gives a more readable
    grouping."""
    def getter(p: ProcessInfo):
        if key == "gpu_index":
            # tuple so reverse=True gives (gpu_index desc, gpu_mem desc);
            # reverse=False gives (gpu_index asc, gpu_mem desc) — always
            # biggest-first within a GPU group.
            return (p.gpu_index, -p.gpu_mem)
        if key == "elapsed_sec":
            return _parse_etime(p.elapsed)
        if key == "cmd_short":
            return p.cmd_short.lower()
        if key == "user":
            return p.user.lower()
        return getattr(p, key, 0)
    return sorted(procs, key=getter, reverse=desc)


# ── nvidia-smi queries ──────────────────────────────────────────────────────

# Per-GPU-handle caches: name/uuid/mem_total/power_limit are effectively
# static, so we query them once. `_gpu_fan_supported` remembers that a card
# doesn't implement NVML_ERROR_NOT_SUPPORTED for fan speed so we skip the
# syscall on the next iteration (saves ~1 ms per missing-fan GPU per tick).
_gpu_static_cache: dict[int, dict] = {}
_gpu_fan_supported: dict[int, bool] = {}
# Slow-changing telemetry (fan, temp, power). We still want them fresh, but
# they don't move 10Hz — refresh every Nth tick and reuse the last reading
# otherwise. On a multi-GPU server this is the single biggest per-iter saving
# because GetFanSpeed / GetPowerUsage each cost ~0.5–1 ms per card.
_SLOW_NVML_STRIDE = 3   # fan / temp / power refresh every Nth tick. These
                        # move glacially (seconds), so a 3 s lag is invisible
                        # while cutting ~2/3 of their per-tick cost.
_gpu_slow_cache: dict[int, tuple[int, int, float]] = {}  # {i: (fan, temp, power_draw_W)}
_slow_nvml_tick = 0


_gpu_mem_cache: dict[int, tuple[int, int]] = {}  # {i: (mem_used_mib, mem_free_mib)}
_MEM_NVML_STRIDE = 1   # every tick (no staleness)


def _query_one_gpu(i: int, refresh_slow: bool, refresh_mem: bool) -> GpuInfo | None:
    """Query a single GPU. Static info is cached forever; fan/temp/power are
    refreshed every `_SLOW_NVML_STRIDE` ticks; memory-info every
    `_MEM_NVML_STRIDE` ticks; util is always fresh (it's what users watch)."""
    try:
        h = _nvml_handle(i)
    except Exception:
        return None
    static = _gpu_static_cache.get(i)
    if static is None:
        try:
            uuid = _decode_str(_nvml.nvmlDeviceGetUUID(h))
            name = _decode_str(_nvml.nvmlDeviceGetName(h))
            mem_total = _nvml.nvmlDeviceGetMemoryInfo(h).total // 1024 // 1024
        except Exception:
            return None
        try:
            power_limit_mw = _nvml.nvmlDeviceGetEnforcedPowerLimit(h)
        except Exception:
            power_limit_mw = 0
        static = {
            "uuid": uuid, "name": name,
            "mem_total": mem_total, "power_limit": power_limit_mw / 1000.0,
        }
        _gpu_static_cache[i] = static
    try:
        util = _nvml.nvmlDeviceGetUtilizationRates(h)
    except Exception:
        return None

    mem_cached = _gpu_mem_cache.get(i)
    if refresh_mem or mem_cached is None:
        try:
            mem = _nvml.nvmlDeviceGetMemoryInfo(h)
            mem_used = mem.used // 1024 // 1024
            mem_free = mem.free // 1024 // 1024
            _gpu_mem_cache[i] = (mem_used, mem_free)
        except Exception:
            if mem_cached:
                mem_used, mem_free = mem_cached
            else:
                mem_used = mem_free = 0
    else:
        mem_used, mem_free = mem_cached

    slow = _gpu_slow_cache.get(i)
    if refresh_slow or slow is None:
        try:
            temp = int(_nvml.nvmlDeviceGetTemperature(h, _nvml.NVML_TEMPERATURE_GPU))
        except Exception:
            temp = slow[1] if slow else 0
        try:
            power_w = _nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            power_w = slow[2] if slow else 0.0
        fan = 0
        if _gpu_fan_supported.get(i, True):
            try:
                fan = int(_nvml.nvmlDeviceGetFanSpeed(h))
            except Exception:
                _gpu_fan_supported[i] = False
                fan = slow[0] if slow else 0
        _gpu_slow_cache[i] = (fan, temp, power_w)
    else:
        fan, temp, power_w = slow

    return GpuInfo(
        index=i, uuid=static["uuid"], name=static["name"],
        mem_total=static["mem_total"],
        mem_used=mem_used, mem_free=mem_free,
        gpu_util=int(util.gpu), mem_util=int(util.memory),
        temp=temp, power_draw=power_w,
        power_limit=static["power_limit"],
        fan_speed=fan,
    )


def _query_gpus_nvml() -> list[GpuInfo]:
    global _slow_nvml_tick
    try:
        count = _nvml.nvmlDeviceGetCount()
    except Exception:
        return []
    tick = _slow_nvml_tick
    _slow_nvml_tick += 1
    refresh_slow = (tick % _SLOW_NVML_STRIDE == 0)
    refresh_mem = (tick % _MEM_NVML_STRIDE == 0)
    gpus: list[GpuInfo] = []
    for i in range(count):
        g = _query_one_gpu(i, refresh_slow, refresh_mem)
        if g is not None:
            gpus.append(g)
    return gpus


def _query_gpus_smi() -> list[GpuInfo]:
    result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,"
         "utilization.gpu,utilization.memory,temperature.gpu,"
         "power.draw,power.limit,fan.speed",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    gpus = []
    for row in csv.reader(io.StringIO(result.stdout.strip())):
        if len(row) < 12:
            continue
        r = [v.strip() for v in row]
        gpus.append(GpuInfo(
            index=int(r[0]), uuid=r[1], name=r[2],
            mem_total=int(r[3]), mem_used=int(r[4]), mem_free=int(r[5]),
            gpu_util=_safe_int(r[6]), mem_util=_safe_int(r[7]),
            temp=_safe_int(r[8]),
            power_draw=_safe_float(r[9]), power_limit=_safe_float(r[10]),
            fan_speed=_safe_int(r[11]),
        ))
    return gpus


def query_gpus() -> list[GpuInfo]:
    if _nvml_init():
        try:
            return _query_gpus_nvml()
        except Exception:
            pass  # fall through to subprocess
    return _query_gpus_smi()


_proc_list_cache: tuple[list[tuple[int, int, str]], float] | None = None
_PROC_LIST_TTL = 3.5   # seconds. Compute-proc enumeration is the single
                       # most expensive NVML call per tick; at the 1 s
                       # default refresh this TTL yields one fresh sample
                       # every ~4 ticks. Running-proc set changes on
                       # process start/exit, which is a human-scale event —
                       # ≤ 3.5 s lag is invisible in practice.


def _query_one_gpu_procs(i: int) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    try:
        h = _nvml_handle(i)
        uuid = _decode_str(_nvml.nvmlDeviceGetUUID(h))
        running = _nvml.nvmlDeviceGetComputeRunningProcesses(h)
        for pi in running:
            used = pi.usedGpuMemory or 0
            used_mib = used // 1024 // 1024 if used else 0
            out.append((int(pi.pid), used_mib, uuid))
    except Exception:
        pass
    return out


def _query_processes_nvml() -> list[tuple[int, int, str]]:
    # NVML compute-process enumeration is the single most expensive call in
    # a steady-state tick (~0.5–1 ms per GPU). Two tricks:
    #  1. cache the result for a short TTL (set changes slowly),
    #  2. when the cache misses on a multi-GPU host, fan out one thread per
    #     GPU since pynvml releases the GIL inside the C call — serial
    #     latency of N × 1 ms collapses to ~ 1.5 ms.
    global _proc_list_cache
    now = time.time()
    if _proc_list_cache is not None:
        cached, ts = _proc_list_cache
        if now - ts < _PROC_LIST_TTL:
            return cached

    try:
        count = _nvml.nvmlDeviceGetCount()
    except Exception:
        count = 0

    procs: list[tuple[int, int, str]] = []
    for i in range(count):
        procs.extend(_query_one_gpu_procs(i))
    _proc_list_cache = (procs, now)
    return procs


def _query_processes_smi() -> list[tuple[int, int, str]]:
    result = subprocess.run(
        ["nvidia-smi",
         "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    procs = []
    for row in csv.reader(io.StringIO(result.stdout.strip())):
        if len(row) < 3:
            continue
        r = [v.strip() for v in row]
        procs.append((int(r[0]), int(r[1]), r[2]))
    return procs


def query_processes() -> list[tuple[int, int, str]]:
    """Returns list of (pid, gpu_mem_mib, gpu_uuid)."""
    if _nvml_init():
        try:
            return _query_processes_nvml()
        except Exception:
            pass
    return _query_processes_smi()


# ── psutil sampling (system + per-process CPU%) ──────────────────────────────

_cpu_primed = False
_proc_cache: dict[int, psutil.Process] = {}


def _prime_and_sample(pids: list[int], sample_interval: float = 0.3) -> None:
    """First call primes psutil and sleeps so subsequent cpu_percent() calls
    return a real delta. On later calls this is a no-op — the time between
    watch-mode iterations serves as the sample window."""
    global _cpu_primed
    if _cpu_primed:
        # Still ensure new PIDs are primed.
        _ensure_proc_primed(pids)
        return
    psutil.cpu_percent(percpu=True)  # prime system
    _ensure_proc_primed(pids)
    time.sleep(sample_interval)
    _cpu_primed = True


def _ensure_proc_primed(pids: list[int]) -> None:
    for pid in pids:
        if pid in _proc_cache:
            continue
        try:
            p = psutil.Process(pid)
            p.cpu_percent()  # first call returns 0.0, primes the delta
            _proc_cache[pid] = p
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
            pass


def _cleanup_proc_cache(alive_pids: set[int]) -> None:
    for pid in list(_proc_cache.keys()):
        if pid not in alive_pids:
            _proc_cache.pop(pid, None)
            _proc_static_cache.pop(pid, None)


_CPU_STATIC_CACHE: tuple[str, int, int] | None = None


def _cpu_static_info() -> tuple[str, int, int]:
    """(model_name, logical_count, physical_count) — cached; values are stable
    for the lifetime of the process."""
    global _CPU_STATIC_CACHE
    if _CPU_STATIC_CACHE is not None:
        return _CPU_STATIC_CACHE
    model = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    # Collapse common noisy substrings ("(R)", "(TM)", "CPU @ ...GHz").
    if model:
        import re as _re
        model = _re.sub(r"\(R\)|\(TM\)|\(tm\)|\(r\)|CPU", "", model)
        model = _re.sub(r"\s+@\s*[\d.]+\s*GHz", "", model)
        model = " ".join(model.split())
    try:
        logical = psutil.cpu_count(logical=True) or 0
        physical = psutil.cpu_count(logical=False) or 0
    except Exception:
        logical = physical = 0
    _CPU_STATIC_CACHE = (model, logical, physical)
    return _CPU_STATIC_CACHE


_BOOT_TIME: float | None = None


def query_system() -> SystemInfo:
    global _BOOT_TIME
    # cpu_total-only: we never display per-core bars, so skip percpu=True
    # (saves ~1 call × N_cores reads of /proc/stat fields per tick).
    cpu_total = psutil.cpu_percent(percpu=False)
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    if _BOOT_TIME is None:
        try:
            _BOOT_TIME = psutil.boot_time()
        except Exception:
            _BOOT_TIME = time.time()
    try:
        load = os.getloadavg()
    except (OSError, AttributeError):
        load = (0.0, 0.0, 0.0)
    # boot_time is static for the life of the kernel; cached once.
    uptime = max(0.0, time.time() - _BOOT_TIME)
    cpu_model, n_logical, n_physical = _cpu_static_info()
    return SystemInfo(
        cpu_per_core=[],  # unused downstream; kept for dataclass compat
        cpu_total=cpu_total,
        load_avg=tuple(load),
        mem_total=vm.total,
        mem_used=vm.used,
        mem_available=vm.available,
        mem_percent=vm.percent,
        swap_total=sw.total,
        swap_used=sw.used,
        swap_percent=sw.percent,
        uptime_seconds=uptime,
        cpu_model=cpu_model,
        cpu_count_logical=n_logical,
        cpu_count_physical=n_physical,
    )


def query_proc_metrics(pids: list[int]) -> dict[int, tuple[float, int]]:
    """Returns {pid: (cpu_percent, rss_bytes)} for alive PIDs."""
    out: dict[int, tuple[float, int]] = {}
    for pid in pids:
        p = _proc_cache.get(pid)
        if p is None:
            continue
        try:
            out[pid] = (p.cpu_percent(), p.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            out[pid] = (0.0, 0)
    return out


def _secs_to_etime(secs: float) -> str:
    """Format seconds as ps-style etime: [D-]HH:MM:SS or MM:SS."""
    secs = max(0, int(secs))
    d = secs // 86400
    h = (secs % 86400) // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if d > 0:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def query_top_procs(sort_key: str, limit: int = 50) -> list[ProcessInfo]:
    """Iterate all system processes and return top N by the given sort key
    (either 'cpu_percent' or 'rss_bytes'). Uses the same psutil Process cache
    as GPU procs so cpu_percent deltas remain accurate across iterations.
    First-seen PIDs report 0% CPU until the next call."""
    rows: list[tuple[float, int, int, psutil.Process]] = []

    for p in psutil.process_iter():
        try:
            pid = p.pid
            if pid not in _proc_cache:
                _proc_cache[pid] = p
                try:
                    p.cpu_percent()  # prime, returns 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                cpu_pct = 0.0
            else:
                try:
                    cpu_pct = _proc_cache[pid].cpu_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            try:
                mem = p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            rows.append((cpu_pct, mem, pid, p))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if sort_key == "rss_bytes":
        rows.sort(key=lambda x: -x[1])
    else:
        rows.sort(key=lambda x: -x[0])

    now_ts = time.time()
    infos: list[ProcessInfo] = []
    for cpu_pct, mem, pid, p in rows[:limit]:
        try:
            with p.oneshot():
                user = p.username() or ""
                try:
                    cmdline = p.cmdline()
                    cmd = " ".join(cmdline) if cmdline else p.name()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    cmd = p.name() if False else ""
                try:
                    create_time = p.create_time()
                    elapsed = _secs_to_etime(now_ts - create_time)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    elapsed = ""
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

        infos.append(ProcessInfo(
            pid=pid,
            gpu_index=-1,
            gpu_mem=0,
            user=user,
            cmd=cmd,
            cmd_short=_shorten_cmd(cmd) if cmd else "",
            cwd="",
            elapsed=elapsed,
            cpu_percent=cpu_pct,
            rss_bytes=mem,
        ))
    return infos


def _fmt_bytes(b: int) -> str:
    for unit, div in (("TiB", 1024 ** 4), ("GiB", 1024 ** 3),
                      ("MiB", 1024 ** 2), ("KiB", 1024)):
        if b >= div:
            return f"{b / div:.1f} {unit}"
    return f"{b} B"


def _fmt_bytes_compact(b: int) -> str:
    """Compact form for tight columns: '1.2G', '234M', '12K'."""
    if b >= 1024 ** 3:
        return f"{b / 1024 ** 3:.1f}G"
    if b >= 1024 ** 2:
        return f"{b / 1024 ** 2:.0f}M"
    if b >= 1024:
        return f"{b / 1024:.0f}K"
    return f"{b}B"


def _fmt_mib(mib: int) -> str:
    """Compact VRAM display: '7.5G' / '24G' / '512M'."""
    if mib >= 1024:
        return f"{mib / 1024:.1f}G"
    return f"{mib}M"


def _fmt_uptime(secs: float) -> str:
    d = int(secs // 86400)
    h = int((secs % 86400) // 3600)
    m = int((secs % 3600) // 60)
    if d > 0:
        return f"{d}d {h}h"
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


# ── Docker container detection ───────────────────────────────────────────────

_container_cache: dict[str, dict] = {}
_has_docker: bool | None = None


def _docker_available() -> bool:
    global _has_docker
    if _has_docker is None:
        _has_docker = shutil.which("docker") is not None
    return _has_docker


def _detect_container(pid: int) -> str | None:
    """Detect if PID runs inside a Docker/Podman container. Returns container ID or None."""
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text()
        # cgroup v2: docker-<id>.scope  (systemd slice)
        m = re.search(r'docker-([a-f0-9]{12,64})\.scope', cgroup)
        if m:
            return m.group(1)[:12]
        # cgroup v1: /docker/<id>
        m = re.search(r'/docker/([a-f0-9]{12,64})', cgroup)
        if m:
            return m.group(1)[:12]
        # containerd / k8s: /cri-containerd-<id>
        m = re.search(r'cri-containerd-([a-f0-9]{12,64})', cgroup)
        if m:
            return m.group(1)[:12]
        # podman
        m = re.search(r'/libpod-([a-f0-9]{12,64})', cgroup)
        if m:
            return m.group(1)[:12]
    except Exception:
        pass
    return None


def _get_container_info(container_id: str) -> dict:
    """Get container name, image, and overlay MergedDir via docker inspect."""
    if container_id in _container_cache:
        return _container_cache[container_id]

    info: dict = {"name": container_id, "image": "", "merged_dir": "", "working_dir": ""}

    if not _docker_available():
        _container_cache[container_id] = info
        return info

    try:
        result = subprocess.run(
            ["docker", "inspect", "--format",
             "{{.Name}}|{{.GraphDriver.Data.MergedDir}}|{{.Config.Image}}|{{.Config.WorkingDir}}",
             container_id],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split("|")
            if len(parts) >= 1 and parts[0]:
                info["name"] = parts[0].lstrip("/")
            if len(parts) >= 2:
                info["merged_dir"] = parts[1]
            if len(parts) >= 3:
                info["image"] = parts[2]
            if len(parts) >= 4:
                info["working_dir"] = parts[3]
    except Exception:
        pass

    _container_cache[container_id] = info
    return info


def _container_cwd(pid: int, container_id: str, cinfo: dict) -> str:
    """Get the container-internal cwd for a process, trying multiple methods."""
    # Method 1: readlink from host — may get overlay path or may fail (permission)
    try:
        host_cwd = os.readlink(f"/proc/{pid}/cwd")
        merged = cinfo.get("merged_dir", "")
        if merged and host_cwd.startswith(merged):
            return host_cwd[len(merged):] or "/"
        return host_cwd
    except PermissionError:
        pass
    except Exception:
        return "(unknown)"

    # Method 2: nsenter into mount namespace (works when running as root)
    try:
        result = subprocess.run(
            ["nsenter", "-m", "-t", str(pid), "readlink", f"/proc/{pid}/cwd"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # Method 3: fallback to container's configured WorkingDir
    wd = cinfo.get("working_dir", "")
    if wd:
        return wd

    return "(container — cwd not accessible)"


# ── Process detail gathering ─────────────────────────────────────────────────

def _detect_parent_session(pid: int) -> str:
    """Walk up process tree via /proc to find tmux/screen. No subprocess needed."""
    try:
        current = pid
        for _ in range(20):
            stat = Path(f"/proc/{current}/stat").read_text()
            comm_start = stat.index("(")
            comm_end = stat.rindex(")")
            comm = stat[comm_start + 1 : comm_end].lower()
            if "tmux" in comm:
                return "tmux"
            if "screen" in comm:
                return "screen"
            ppid = int(stat[comm_end + 2 :].split()[1])
            if ppid <= 1:
                break
            current = ppid
    except Exception:
        pass
    return ""


# Per-PID static data cache: cmd/cwd/conda/venv/container/parent_info do
# not change for the lifetime of a process, so we compute them once per
# PID and reuse. On cache hit, get_process_info() avoids 4+ /proc reads,
# a cgroup regex scan, docker cgroup match, and a 20-deep tmux ancestry
# walk. (elapsed is recomputed each tick from cached create_time.)
_proc_static_cache: dict[int, dict] = {}


def _get_proc_static(pid: int, psp: psutil.Process | None) -> dict:
    """Return (and lazily populate) the static-per-PID bundle used by
    get_process_info. Returns keys: cmd, cmd_short, cwd, conda_env, venv,
    container_id, container_name, container_image, parent_info,
    create_time, user."""
    cached = _proc_static_cache.get(pid)
    if cached is not None:
        return cached

    info: dict = {
        "cmd": "", "cmd_short": "", "cwd": "",
        "conda_env": "", "venv": "",
        "container_id": None, "container_name": "", "container_image": "",
        "parent_info": "", "create_time": 0.0, "user": "",
    }

    # user + create_time via psutil (one-shot).
    if psp is not None:
        try:
            with psp.oneshot():
                try:
                    info["user"] = psp.username() or ""
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                try:
                    info["create_time"] = psp.create_time()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # cmdline
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        cmd = raw.replace(b'\x00', b' ').decode('utf-8', errors='replace').strip()
        cmd = re.sub(r'[\r\n\t]+', ' ', cmd)
        info["cmd"] = cmd
        info["cmd_short"] = _shorten_cmd(cmd)
    except Exception:
        info["cmd"] = "(access denied)"
        info["cmd_short"] = info["cmd"]

    # cwd
    try:
        info["cwd"] = os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        info["cwd"] = "(access denied)"

    # docker/podman container
    container_id = _detect_container(pid)
    if container_id:
        cinfo = _get_container_info(container_id)
        info["container_id"] = container_id
        info["container_name"] = cinfo.get("name", container_id)
        info["container_image"] = cinfo.get("image", "")
        info["cwd"] = _container_cwd(pid, container_id, cinfo)

    # conda / venv
    try:
        env_raw = Path(f"/proc/{pid}/environ").read_bytes()
        for var in env_raw.split(b'\x00'):
            decoded = var.decode('utf-8', errors='replace')
            if decoded.startswith("CONDA_DEFAULT_ENV="):
                val = decoded.split("=", 1)[1]
                if val and val != "base":
                    info["conda_env"] = val
            elif decoded.startswith("VIRTUAL_ENV="):
                info["venv"] = decoded.split("=", 1)[1]
    except Exception:
        pass

    # tmux/screen parent walk
    info["parent_info"] = _detect_parent_session(pid)

    _proc_static_cache[pid] = info
    return info


def get_process_info(pid: int, gpu_index: int, gpu_mem: int) -> ProcessInfo:
    p = ProcessInfo(pid=pid, gpu_index=gpu_index, gpu_mem=gpu_mem)

    psp: psutil.Process | None = None
    try:
        if pid not in _proc_cache:
            _proc_cache[pid] = psutil.Process(pid)
        psp = _proc_cache[pid]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        psp = None

    static = _get_proc_static(pid, psp)
    p.user = static["user"]
    p.cmd = static["cmd"]
    p.cmd_short = static["cmd_short"]
    p.cwd = static["cwd"]
    p.conda_env = static["conda_env"]
    p.venv = static["venv"]
    p.container_name = static["container_name"]
    p.container_image = static["container_image"]
    p.parent_info = static["parent_info"]

    ct = static["create_time"]
    if ct:
        now = time.time()
        p.elapsed = _secs_to_etime(now - ct)
        p.started = datetime.fromtimestamp(ct).strftime("%a %b %d %H:%M:%S %Y")

    return p


def _shorten_cmd(cmd: str) -> str:
    cmd = re.sub(r'/home/([^/\s]+)', lambda m: '~', cmd)
    cmd = re.sub(r'\S+/bin/python[23]?\s', 'python ', cmd)
    # Collapse embedded newlines / tabs / CR into single spaces so
    # `python -c "line1\nline2"` style commands don't break the box layout.
    cmd = re.sub(r'[\r\n\t]+', ' ', cmd)
    return cmd


def _trunc(text: str, maxlen: int) -> str:
    if maxlen < 4:
        return text[:maxlen]
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 3] + "..."


# ── Rendering primitives ─────────────────────────────────────────────────────

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def _vlen(s: str) -> int:
    """Visible length — strips ANSI escape codes."""
    # Regex (C-implemented) is ~5x faster here than a hand-rolled Python loop
    # in realistic workloads: most strings only contain a handful of escapes,
    # and re.sub's C loop pays far less per-character cost than the Python
    # interpreter's `while i < L: s[i]` scan.
    return len(_ANSI_RE.sub('', s))


def _ansi_truncate(s: str, max_visible: int, ellipsis: str = "…") -> str:
    """Truncate to `max_visible` visible chars while preserving ANSI escape
    codes. Without this, naive slicing strips color mid-sequence and causes
    the rest of the row (and subsequent rows) to lose formatting."""
    if _vlen(s) <= max_visible:
        return s
    target = max(0, max_visible - len(ellipsis))
    out: list[str] = []
    visible = 0
    i = 0
    n = len(s)
    while i < n:
        if s[i] == '\x1b':
            m = _ANSI_RE.match(s, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
        if visible >= target:
            break
        out.append(s[i])
        visible += 1
        i += 1
    out.append(ellipsis)
    if _COLOR_ON:
        out.append(C.RESET)
    return "".join(out)


def _rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


# btop default-ish gradient stops: green → yellow → red
_GRAD_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.0, (0x50, 0xF0, 0x95)),
    (0.5, (0xF2, 0xE2, 0x60)),
    (1.0, (0xEE, 0x44, 0x45)),
]


def _grad_ansi(pct_0_100: float) -> str:
    """Return an ANSI truecolor escape for the given percentage on the
    green→yellow→red ramp, or empty string when color is disabled."""
    if not _COLOR_ON:
        return ""
    r, g, b = _grad_color(max(0.0, min(1.0, pct_0_100 / 100.0)))
    return _rgb(r, g, b)


def _reset() -> str:
    return C.RESET if _COLOR_ON else ""


# Precomputed gradient LUT — 1001 entries gives ≤ 0.1% step granularity across
# [0, 1], finer than any terminal cell and indistinguishable from the on-the-
# fly interpolation. This is the hottest function in the render loop
# (thousands of calls per tick for bar rendering).
_GRAD_LUT_SIZE = 100001   # 1e-5 step — finer than int-RGB truncation, so the
                           # LUT is bit-identical to the on-the-fly formula.
_GRAD_LUT: list[tuple[int, int, int]] = []


def _build_grad_lut() -> None:
    out = []
    stops = _GRAD_STOPS
    N = _GRAD_LUT_SIZE
    for idx in range(N):
        t = idx / (N - 1)
        color = stops[-1][1]
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t0 <= t <= t1:
                k = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                color = (
                    int(c0[0] + k * (c1[0] - c0[0])),
                    int(c0[1] + k * (c1[1] - c0[1])),
                    int(c0[2] + k * (c1[2] - c0[2])),
                )
                break
        out.append(color)
    _GRAD_LUT[:] = out


_build_grad_lut()


def _grad_color(t: float) -> tuple[int, int, int]:
    if t <= 0.0:
        return _GRAD_LUT[0]
    if t >= 1.0:
        return _GRAD_LUT[-1]
    # round-to-nearest, not truncate — picks the LUT bucket whose t matches
    # the caller's t up to LUT resolution. With size=100001 this makes the
    # returned RGB identical to the int-truncated linear interpolation.
    return _GRAD_LUT[int(t * (_GRAD_LUT_SIZE - 1) + 0.5)]


# Box border color (soft blue, like btop's panel accent)
_BORDER_RAW = "\033[38;2;90;130;200m"
# Empty bar cell color (dim slate)
_EMPTY_RAW = "\033[38;2;60;60;80m"
# Button text color (for clickable box titles) — bright neon cyan, bold.
# The surrounding `┤ ├` of the box top already frames the title, so we
# don't add extra brackets or fill (overline support is too patchy across
# terminals to draw a top edge reliably).
_BTN_FG_RAW = "\033[38;2;0;214;255m\033[1m"
# Panel background — a very dark blue-gray so boxes read as sunken panels
# like btop's theme. Applied via post-process so existing fg/color rendering
# doesn't need to be touched.
_PANEL_BG_RAW = "\033[48;2;15;15;25m"


def _apply_row_bg(line: str, width: int, bg_raw: str) -> str:
    """Wrap an already-rendered line with a solid background color, and
    re-inject the bg after every RESET so fg color changes inside the line
    don't blow away the bg. Also pads with spaces (on bg) up to `width`."""
    if not _COLOR_ON:
        return line
    vlen = _vlen(line)
    pad = " " * max(0, width - vlen)
    with_bg = line.replace("\x1b[0m", f"\x1b[0m{bg_raw}")
    return f"{bg_raw}{with_bg}{pad}\x1b[0m"


def _apply_panel_bg(line: str, width: int) -> str:
    return _apply_row_bg(line, width, _PANEL_BG_RAW)


def _border(s: str) -> str:
    """Wrap border glyphs in soft-blue color when color is enabled."""
    if _COLOR_ON:
        return f"{_BORDER_RAW}{s}{C.RESET}"
    return s


def _btn(label: str, key: str | None = None) -> str:
    """Render text as a clickable-looking title — bright neon-cyan bold,
    ▶ arrow prefix. The shortcut for each title is documented in the
    footer hint, not inline."""
    if _COLOR_ON:
        return f"{_BTN_FG_RAW} ▶ {label} {C.RESET}"
    return f" ▶ {label} "


_TRACK_RAW = "\033[38;2;150;150;170m"  # bright enough to frame the bar track


def bar(pct: float, width: int = 30) -> str:
    """btop-style gradient bar wrapped in ▕ ▏ track markers. The
    markers are always visible (even at 100% fill), so the right-hand
    limit of the bar reads at a glance. Body cells use slim ■ for
    filled and dim · for empty."""
    if not _COLOR_ON:
        filled = max(0, min(width, int(pct / 100 * width)))
        return "#" * filled + "-" * (width - filled)

    if width < 4:
        # Too narrow for brackets — plain rendering.
        frac = max(0.0, min(1.0, pct / 100.0))
        full = max(0, min(width, round(frac * width)))
        parts = []
        denom = max(1, width - 1)
        for i in range(full):
            r, g, b = _grad_color(i / denom)
            parts.append(f"{_rgb(r, g, b)}■")
        if width - full > 0:
            parts.append(_EMPTY_RAW + "·" * (width - full))
        parts.append(C.RESET)
        return "".join(parts)

    inner_w = width - 2
    frac = max(0.0, min(1.0, pct / 100.0))
    full = max(0, min(inner_w, round(frac * inner_w)))

    parts: list[str] = [f"{_TRACK_RAW}▕"]
    denom = max(1, inner_w - 1)
    for i in range(full):
        r, g, b = _grad_color(i / denom)
        parts.append(f"{_rgb(r, g, b)}■")

    empty = inner_w - full
    if empty > 0:
        parts.append(_EMPTY_RAW + "·" * empty)
    parts.append(f"{_TRACK_RAW}▏{C.RESET}")
    return "".join(parts)


# ── Box drawing ──────────────────────────────────────────────────────────────

_TITLE_KEYS = {
    "cpu":       "c",
    "memory":    "m",
    "gpus":      "g",
    "processes": "p",
}


def _box_top(title: str, width: int, subtitle: str = "",
             clickable: bool = False, subtitle_dim: bool = True) -> str:
    """╭──┤ [title] ├────────── subtitle ──╮
    When clickable=True the title is rendered as a button (highlighted
    background) so users see it as a click target. subtitle_dim=False
    renders the subtitle in normal (non-dim) color — used for boxes whose
    subtitle carries info we want visible at a glance (e.g. GPU model)."""
    title_rendered = _btn(title, key=_TITLE_KEYS.get(title)) if clickable else f" {cb(title)} "
    prefix = f"{_border('╭──┤')}{title_rendered}{_border('├')}"
    if subtitle:
        # Truncate subtitle so it never pushes the right border off the line.
        # Budget: width - prefix - borders("┤  ├──╮" = 7 visible) = max subtitle width.
        budget = width - _vlen(prefix) - 7
        if budget < 1:
            suffix = _border("──╮")
        else:
            vis_sub = subtitle if _vlen(subtitle) <= budget \
                else _ansi_truncate(subtitle, budget)
            styled = c(C.DIM, vis_sub) if subtitle_dim else vis_sub
            suffix = f"{_border('┤')} {styled} {_border('├──╮')}"
    else:
        suffix = _border("──╮")
    fill = width - _vlen(prefix) - _vlen(suffix)
    return prefix + _border("─" * max(0, fill)) + suffix


def _box_bottom(width: int) -> str:
    return _border("╰" + "─" * (width - 2) + "╯")


def _box_line(content: str, width: int) -> str:
    """│ content (padded to width-4) │"""
    inner = width - 4
    vlen = _vlen(content)
    if vlen > inner:
        content = _ansi_truncate(content, inner)
        vlen = _vlen(content)
    pad = inner - vlen
    return f"{_border('│')} {content}{' ' * pad} {_border('│')}"


# ── Boxed section renderers ──────────────────────────────────────────────────

def _cpu_subtitle(sysinfo: SystemInfo, budget: int) -> str:
    """Build CPU box subtitle, shrinking gracefully to fit `budget` visible
    columns. Richest form: '12% · 24 cores · AMD EPYC 7763'."""
    pct = f"{sysinfo.cpu_total:.0f}%"
    nL = sysinfo.cpu_count_logical
    model = sysinfo.cpu_model
    candidates = []
    if nL and model:
        candidates.append(f"{pct} · {nL} cores · {model}")
    if nL:
        candidates.append(f"{pct} · {nL} cores")
    candidates.append(pct)
    for s in candidates:
        if len(s) <= budget:
            return s
    return pct


def render_cpu_box(sysinfo: SystemInfo, W: int) -> list[str]:
    """CPU box: single usage bar + percent. 4 lines (one blank row to
    keep the height matched with the memory box for hstack alignment)."""
    lines: list[str] = []
    # Budget roughly matches _box_top's subtitle budget (prefix + borders = ~20
    # for "cpu" clickable). Use a conservative value so we don't get truncated.
    sub_budget = max(4, W - 20)
    subtitle = _cpu_subtitle(sysinfo, sub_budget)
    lines.append(_box_top("cpu", W, subtitle, clickable=True, subtitle_dim=False))
    inner_w = W - 4

    bar_w = max(8, inner_w - 8)
    pct_plain = f"{sysinfo.cpu_total:>5.1f}%"   # always 6 chars
    if _COLOR_ON:
        r, g, b = _grad_color(sysinfo.cpu_total / 100)
        pct_s = f"{_rgb(r, g, b)}{pct_plain}{C.RESET}"
    else:
        pct_s = pct_plain
    lines.append(_box_line(f"{bar(sysinfo.cpu_total, bar_w)} {pct_s}", W))
    lines.append(_box_bottom(W))
    return lines


def _mem_subtitle(sysinfo: SystemInfo, budget: int) -> str:
    """Memory box subtitle with swap usage. Richest form:
    '35% · swap 2.3G / 8.0G (66%)'. Shrinks to just memory % when tight."""
    mem_pct = f"{sysinfo.mem_percent:.0f}%"
    if sysinfo.swap_total <= 0:
        return mem_pct
    sp = sysinfo.swap_percent
    su_g = sysinfo.swap_used / 1024 ** 3
    st_g = sysinfo.swap_total / 1024 ** 3
    candidates = [
        f"{mem_pct} · swap {su_g:.1f}G / {st_g:.1f}G ({sp:.0f}%)",
        f"{mem_pct} · swap {su_g:.1f}G / {st_g:.1f}G",
        f"{mem_pct} · swap {sp:.0f}%",
        mem_pct,
    ]
    for s in candidates:
        if len(s) <= budget:
            return s
    return mem_pct


def render_mem_box(sysinfo: SystemInfo, W: int) -> list[str]:
    """Memory box: single Mem usage row. 3 lines — Swap moves to the subtitle."""
    lines: list[str] = []
    sub_budget = max(4, W - 23)   # matches _box_top budget for "memory" title
    subtitle = _mem_subtitle(sysinfo, sub_budget)
    lines.append(_box_top("memory", W, subtitle, clickable=True, subtitle_dim=False))
    inner_w = W - 4

    # Fixed budget: label(4) + "  " + " " + mem_txt(16) + "  " + pct(4)
    MEM_TXT_W = 16
    fixed = 4 + 2 + 1 + MEM_TXT_W + 2 + 4
    bar_w = max(6, inner_w - fixed)

    def _row(label: str, pct: float, used: int, total: int) -> str:
        used_s = f"{used / 1024 ** 3:.1f}G"
        total_s = f"{total / 1024 ** 3:.1f}G"
        mem_txt = f"{used_s} / {total_s}".rjust(MEM_TXT_W)
        return (
            f"{cb(label.ljust(4))}  {bar(pct, bar_w)} "
            f"{mem_txt}  {c(C.DIM, f'{pct:>3.0f}%')}"
        )

    lines.append(_box_line(
        _row("Mem", sysinfo.mem_percent, sysinfo.mem_used, sysinfo.mem_total), W))
    lines.append(_box_bottom(W))
    return lines


def render_cpu_mem_compact(sysinfo: SystemInfo, W: int) -> list[str]:
    """Single-line compact CPU + Memory summaries without borders.
    Used on short terminals where boxed versions would push content off-screen.
    Returns exactly 2 lines (row 0 = cpu, row 1 = memory)."""
    reset = _reset()
    # CPU line: [cpu] NN.N%  bar  load a b c  up Xd Xh
    bar_w = max(10, W - 55)
    cpu_pct = sysinfo.cpu_total
    cpu_col = _grad_ansi(cpu_pct)
    load = sysinfo.load_avg
    cpu_line = (
        f" {_btn('cpu')}  {cpu_col}{cpu_pct:>5.1f}%{reset}  "
        f"{bar(cpu_pct, bar_w)}  "
        f"{c(C.DIM, f'load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}')}"
        f"  {c(C.DIM, f'up {_fmt_uptime(sysinfo.uptime_seconds)}')}"
    )

    # Memory line: [memory] NN.N%  bar  used/total [ swap N% ]
    mem_pct = sysinfo.mem_percent
    mem_col = _grad_ansi(mem_pct)
    mem_used_g = sysinfo.mem_used / 1024 ** 3
    mem_total_g = sysinfo.mem_total / 1024 ** 3
    mem_line = (
        f" {_btn('memory')}  {mem_col}{mem_pct:>5.1f}%{reset}  "
        f"{bar(mem_pct, bar_w)}  "
        f"{mem_used_g:>5.1f}G / {mem_total_g:<6.1f}G"
    )
    if sysinfo.swap_total > 0:
        sp = sysinfo.swap_percent
        sp_col = _grad_ansi(sp)
        mem_line += f"  {c(C.DIM, 'swap')} {sp_col}{sp:>3.0f}%{reset}"

    return [cpu_line, mem_line]


def _hstack(left: list[str], right: list[str], left_w: int, right_w: int) -> list[str]:
    """Place two pre-rendered boxes side by side. Pads shorter side with spaces
    so the right box lines up vertically."""
    n = max(len(left), len(right))
    out = []
    for i in range(n):
        l = left[i] if i < len(left) else " " * left_w
        r = right[i] if i < len(right) else " " * right_w
        out.append(l + r)
    return out


def _gpu_content_parts(gpu: GpuInfo) -> tuple[str, str, str, str, str, str, str]:
    """Return (util_pfx, util_sfx, vram_pfx, vram_sfx, power_s, temp_s, fan_s)
    as plain-text strings (no ANSI). All fields use fixed-width formatting so
    every GPU row renders to the same visible length — the right edge of each
    box lines up regardless of whether numbers are 1, 2, or 3 digits."""
    # VRAM: "used / total" with a space on each side of the slash.
    # Right-pad used / left-pad total to 6 chars each → always 15 visible chars.
    vram_text = f"{_fmt_mib(gpu.mem_used):>6} / {_fmt_mib(gpu.mem_total):<6}"
    util_pfx = "Util: "
    util_sfx = f"  {gpu.gpu_util:>3}%"                # always 6 chars
    vram_pfx = "VRAM: "
    vram_sfx = f"  {vram_text}"                       # always 17 chars
    # 4-digit width handles 4-digit watts (B200 / H200 cards push past
    # 1000 W); without this the Power column shifts when a card spikes
    # above 999 W and the in-row separators stop aligning across rows.
    power_s = f"Power: {gpu.power_draw:>4.0f} / {gpu.power_limit:.0f}W"
    temp_s = f"Temp: {gpu.temp:>3d}°C"                # 11 chars
    fan_s = f"Fan: {gpu.fan_speed:>3d}%" if gpu.fan_speed else ""
    return util_pfx, util_sfx, vram_pfx, vram_sfx, power_s, temp_s, fan_s


def _gpu_fixed_non_bar(gpu: GpuInfo, gap: str = "   ") -> int:
    util_pfx, util_sfx, vram_pfx, vram_sfx, power_s, temp_s, fan_s = _gpu_content_parts(gpu)
    meta_parts = [power_s, temp_s] + ([fan_s] if fan_s else [])
    meta_text = gap.join(meta_parts)
    return (
        len(util_pfx) + len(util_sfx) + len(gap)
        + len(vram_pfx) + len(vram_sfx) + len(gap)
        + len(meta_text)
    )


def _unified_bar_w(gpus: list[GpuInfo], inner_w: int) -> int:
    """Return a single bar_w that guarantees the widest GPU still fits.
    Using the SAME bar_w for every GPU keeps bars visually aligned."""
    if not gpus:
        return 10
    worst = max(_gpu_fixed_non_bar(g) for g in gpus)
    return max(3, (inner_w - worst) // 2)


def render_gpus_box(gpus: list[GpuInfo], W: int,
                    state: UIState | None = None,
                    show_separators: bool = False,
                    busy_gpus: set[int] | None = None) -> list[str]:
    """Combined GPU box — one row per GPU inside a single outer box.
    Saves vertical space versus one-box-per-GPU and keeps all bars aligned."""
    lines: list[str] = []
    if not gpus:
        return lines

    # Build subtitle "N × <model>" when all GPUs share the same name,
    # otherwise just "N GPUs".
    names = {g.name for g in gpus}
    if len(names) == 1:
        # The full nvidia-smi name (e.g. "NVIDIA B200") fits comfortably in
        # the title row; no need to strip the vendor prefix.
        subtitle = f"{len(gpus)} × {next(iter(names))}"
    else:
        subtitle = f"{len(gpus)} GPUs"
    lines.append(_box_top("gpus", W, subtitle, clickable=True, subtitle_dim=False))

    inner_w = W - 4
    # Column separator: dim vertical bar between sections, like btop's table.
    col_sep_raw = "│"
    gap = f" {col_sep_raw} "  # " │ " — 3 visible chars
    col_sep = c(C.DIM, col_sep_raw)
    gap_rendered = f" {col_sep} "
    # GPU index width: enough to hold the largest index
    idx_w = max(1, len(str(max(g.index for g in gpus))))
    # Prefix plain width: "GPU "(4) + idx(idx_w)
    prefix_w = 4 + idx_w

    # Unified bar_w across all rows so bars line up for comparison.
    # Row structure: prefix  gap  Util bar util_sfx  gap  VRAM bar vram_sfx  gap  meta_text
    # Adaptive: drop Fan → Temp → Power when the row would otherwise overflow
    # (prefer hiding a whole column over mid-column `…` truncation).
    MIN_BAR = 6

    def _max_fixed(include_power: bool, include_temp: bool, include_fan: bool) -> int:
        m = 0
        for g in gpus:
            _, util_sfx, _, vram_sfx, power_s, temp_s, fan_s = _gpu_content_parts(g)
            meta_parts = []
            if include_power:
                meta_parts.append(power_s)
            if include_temp:
                meta_parts.append(temp_s)
            if include_fan and fan_s:
                meta_parts.append(fan_s)
            meta_text = gap.join(meta_parts)
            fixed = (prefix_w + len(gap)
                     + len("Util: ") + len(util_sfx) + len(gap)
                     + len("VRAM: ") + len(vram_sfx))
            if meta_text:
                fixed += len(gap) + len(meta_text)
            m = max(m, fixed)
        return m

    # Try full → drop fan → drop temp → drop power. Pick the first config whose
    # bar width is readable.
    for inc_p, inc_t, inc_f in [(True, True, True), (True, True, False),
                                 (True, False, False), (False, False, False)]:
        max_fixed = _max_fixed(inc_p, inc_t, inc_f)
        if (inner_w - max_fixed) // 2 >= MIN_BAR:
            break
    include_power, include_temp, include_fan = inc_p, inc_t, inc_f
    bar_w = max(3, (inner_w - max_fixed) // 2)

    # Dotted separator line used between rows for vertical breathing room.
    sep_line = c(C.DIM, "┈" * inner_w)

    reset = _reset()
    for i, gpu in enumerate(gpus):
        _, util_sfx, _, vram_sfx, power_s, temp_s, fan_s = _gpu_content_parts(gpu)
        mem_pct = gpu.mem_used / gpu.mem_total * 100 if gpu.mem_total else 0
        power_pct = (gpu.power_draw / gpu.power_limit * 100) if gpu.power_limit else 0
        tc = C.RED if gpu.temp >= 85 else C.YELLOW if gpu.temp >= 70 else C.GREEN

        util_col = _grad_ansi(gpu.gpu_util)
        vram_col = _grad_ansi(mem_pct)
        power_col = _grad_ansi(power_pct)
        fan_col = _grad_ansi(gpu.fan_speed) if gpu.fan_speed else ""

        # GPU label colour:
        #   * active filter → cyan (takes priority)
        #   * GPU has running procs → red (bar's right-end colour)
        #   * GPU is idle → green (bar's left-end colour)
        filtered = state is not None and state.filter_gpu == gpu.index
        label_text = f"GPU {gpu.index:>{idx_w}}"
        if filtered:
            idx_label = c(C.CYAN + C.BOLD, label_text)
        elif _COLOR_ON:
            has_procs = busy_gpus is not None and gpu.index in busy_gpus
            r, g, b = _grad_color(1.0 if has_procs else 0.0)
            idx_label = f"{_rgb(r, g, b)}{C.BOLD}{label_text}{C.RESET}"
        else:
            idx_label = cb(label_text)

        util_num = util_sfx[2:]     # "NNN%"  (strip leading 2-space pad)
        vram_val = vram_sfx[2:]     # "used/total"

        util_part = f"{cb('Util:')} {bar(gpu.gpu_util, bar_w)} {util_col}{util_num}{reset}"
        vram_part = f"{cb('VRAM:')} {bar(mem_pct, bar_w)} {vram_col}{vram_val}{reset}"
        parts = [util_part, vram_part]
        if include_power:
            parts.append(f"{cb('Power:')} {power_col}{power_s[len('Power: '):]}{reset}")
        if include_temp:
            parts.append(f"{cb('Temp:')} " + c(tc + C.BOLD, temp_s[len('Temp: '):]))
        if include_fan and fan_s:
            parts.append(f"{cb('Fan:')} {fan_col}{fan_s[len('Fan: '):]}{reset}")

        row = idx_label + gap_rendered + gap_rendered.join(parts)
        lines.append(_box_line(row, W))
        if show_separators and i < len(gpus) - 1:
            lines.append(_box_line(sep_line, W))

    lines.append(_box_bottom(W))
    return lines


# Process table column layout. (key, label, width, align).
# Widths include label visible chars only; separators are 2 spaces between cols.
_PROC_COLS_ALL = [
    # (key,         label,      width, align, modes)
    ("gpu_index",   "GPU",       3, "right", ("gpu",)),
    ("pid",         "PID",       7, "right", ("gpu", "cpu", "mem")),
    ("user",        "USER",      5, "left",  ("gpu", "cpu", "mem")),
    ("gpu_mem",     "GPU MEM",  10, "right", ("gpu",)),
    ("cpu_percent", "%CPU",      5, "right", ("cpu", "mem")),
    ("rss_bytes",   "RES",       6, "right", ("cpu", "mem")),
    ("elapsed_sec", "ELAPSED",  10, "right", ("gpu", "cpu", "mem")),
    ("cmd_short",   "COMMAND",   7, "left",  ("gpu", "cpu", "mem")),
]


def _proc_cols_for(mode: str) -> list[tuple[str, str, int, str]]:
    """Pick the visible column set for the current mode (gpu/cpu/mem)."""
    return [(k, l, w, a) for (k, l, w, a, modes) in _PROC_COLS_ALL if mode in modes]


# Back-compat alias used by older callers inside the file.
_PROC_COLS = _proc_cols_for("gpu")
_PROC_COL_GAP = 2
# Left-edge offset of content within each row (_box_line prefix: "│ " = 2 terminal cols before content)
_CONTENT_COL_OFFSET = 3  # 1-indexed terminal col where content[0] sits


def _proc_col_bounds(inner_w: int, mode: str = "gpu") -> list[tuple[int, int, str]]:
    """Return (content_col_start, content_col_end, sort_key) — 0-indexed, inclusive.
    Each column's click region includes the trailing gap; the last col
    (COMMAND) extends to the end of the inner content area."""
    cols = _proc_cols_for(mode)
    bounds = []
    offset = 0
    n = len(cols)
    for i, (key, _label, width, _align) in enumerate(cols):
        is_last = i == n - 1
        if is_last:
            end = max(offset, inner_w - 1)
        else:
            end = offset + width + _PROC_COL_GAP - 1
        bounds.append((offset, end, key))
        offset = end + 1
    return bounds


_SELECTED_BG_RAW = "\033[48;2;40;70;130m"   # row highlight for selected proc


def render_proc_box(all_procs: list[ProcessInfo], W: int,
                    state: UIState | None = None,
                    long: bool = False,
                    hidden_count: int = 0,
                    extra_cmd_rows: int = 0
                    ) -> tuple[list[str], list[tuple[int, int, int]]]:
    """Returns (lines, proc_rows) where proc_rows is a list of
    (local_line_start, local_line_end, pid) — indices into `lines` for
    the row(s) belonging to each rendered process.

    `extra_cmd_rows`: in non-long mode, allow each proc to wrap the command
    up to this many extra rows (distributed from spare vertical space) so the
    display fills the terminal uniformly instead of leaving a big empty gap."""
    lines: list[str] = []
    subtitle_count = f"{len(all_procs)} proc" + ("" if len(all_procs) == 1 else "s") \
        if all_procs else "none"
    mode_label = ""
    if state is not None and state.mode != "gpu":
        metric = {"cpu": "by CPU", "mem": "by MEM"}.get(state.mode, state.mode)
        mode_label = f" · {metric} · click to reset ◀"
    subtitle = subtitle_count + mode_label
    lines.append(_box_top("processes", W, subtitle, clickable=True))

    proc_rows: list[tuple[int, int, int]] = []
    if not all_procs:
        lines.append(_box_line(c(C.DIM, "no GPU processes"), W))
        lines.append(_box_bottom(W))
        return lines, proc_rows

    inner_w = W - 4
    active_key = state.sort_key if state else None
    mode = state.mode if state else "gpu"
    cols = _proc_cols_for(mode)

    # Header: underline only the label glyphs (not padding) so it reads as
    # a link. Active sort column additionally gets cyan highlight.
    UL = "\033[4m"
    hdr_segs = []
    for key, label, width, align in cols:
        color = (C.CYAN + C.BOLD + UL) if key == active_key else (C.BOLD + UL)
        colored = c(color, label)
        pad_n = width - len(label)
        seg = (" " * pad_n + colored) if align == "right" else (colored + " " * pad_n)
        hdr_segs.append(seg)
    hdr = (" " * _PROC_COL_GAP).join(hdr_segs)
    lines.append(_box_line(hdr, W))

    # CMD_COL is the sum of all cols before COMMAND plus gaps
    CMD_COL = 0
    for i, (_k, _l, width, _a) in enumerate(cols):
        if i == len(cols) - 1:
            break
        CMD_COL += width + _PROC_COL_GAP

    cmd_width = max(20, inner_w - CMD_COL)
    selected_pid = state.selected_pid if state else None

    for proc in all_procs:
        proc_start_line = len(lines)
        is_selected = proc.pid == selected_pid
        user_color = _user_color(proc.user)

        cpu_raw = f"{proc.cpu_percent:.0f}" if proc.cpu_percent >= 100 else f"{proc.cpu_percent:.1f}"
        cpu_color = (
            C.RED if proc.cpu_percent >= 200
            else C.YELLOW if proc.cpu_percent >= 50
            else C.GREEN if proc.cpu_percent >= 10
            else C.DIM
        )

        # Build each cell's value based on which cols are active for this mode
        cell: dict[str, str] = {
            "gpu_index":  c(C.WHITE,  f"{proc.gpu_index:>3}" if proc.gpu_index >= 0 else f"{'-':>3}"),
            "pid":        c(C.YELLOW, f"{proc.pid:>7}"),
            "user":       c(user_color, _trunc(proc.user, 5).ljust(5)),
            "gpu_mem":    (f"{proc.gpu_mem:>6,} MiB" if proc.gpu_mem > 0 else f"{'-':>10}"),
            "cpu_percent": c(cpu_color, f"{cpu_raw:>5}"),
            "rss_bytes":  (f"{_fmt_bytes_compact(proc.rss_bytes):>6}" if proc.rss_bytes else f"{'-':>6}"),
            "elapsed_sec": f"{proc.elapsed:>10}",
        }
        # COMMAND column is rendered after prefix, so assemble all non-COMMAND.
        prefix_cells = [cell[k] for (k, _l, _w, _a) in cols if k != "cmd_short"]
        prefix = ("  " * 0) + ("  ").join(prefix_cells) + "  "

        # Build the inline meta (cwd + badges) once — we may append it after
        # the command on the same row to keep each proc compact, instead of
        # burning a whole second row on it.
        meta_parts_colored: list[str] = []
        cwd_disp = proc.cwd.replace(f"/home/{proc.user}", "~") if proc.cwd else ""
        if cwd_disp:
            meta_parts_colored.append(c(C.CYAN, cwd_disp))
        if proc.container_name:
            docker_label = proc.container_name
            if proc.container_image:
                docker_label += f"({proc.container_image})"
            meta_parts_colored.append(c(C.RED, f"[{docker_label}]"))
        if proc.conda_env:
            meta_parts_colored.append(c(C.GREEN, f"[{proc.conda_env}]"))
        if proc.venv:
            meta_parts_colored.append(c(C.MAGENTA, f"[{os.path.basename(proc.venv)}]"))
        if proc.parent_info:
            meta_parts_colored.append(c(C.BLUE, f"[{proc.parent_info}]"))
        meta_inline = " ".join(meta_parts_colored) if meta_parts_colored else ""
        meta_vlen = _vlen(meta_inline)

        cmd_text = proc.cmd_short or ""
        # First-row cmd width (reduced if inline meta is appended on row 1).
        sep = " · "
        if meta_inline:
            cmd_width_first = max(15, cmd_width - meta_vlen - len(sep))
        else:
            cmd_width_first = cmd_width

        if long and cmd_text and len(cmd_text) > cmd_width:
            # Long mode — wrap the full command, meta on its own row at end.
            first = cmd_text[:cmd_width]
            lines.append(_box_line(prefix + first, W))
            remaining = cmd_text[cmd_width:]
            cont_indent = " " * CMD_COL
            while remaining:
                chunk = remaining[:cmd_width]
                remaining = remaining[cmd_width:]
                lines.append(_box_line(cont_indent + chunk, W))
            if meta_inline:
                lines.append(_box_line(cont_indent + meta_inline, W))
        elif extra_cmd_rows > 0 and cmd_text and len(cmd_text) > cmd_width_first:
            # Compact + spare rows: meta stays inline on row 1, cmd wraps into
            # `extra_cmd_rows` continuation rows under the COMMAND column so
            # the proc box fills vertical space without wasting rows.
            first = cmd_text[:cmd_width_first]
            row1 = prefix + first
            if meta_inline:
                row1 += c(C.DIM, sep) + meta_inline
            lines.append(_box_line(row1, W))
            remaining = cmd_text[cmd_width_first:]
            cont_indent = " " * CMD_COL
            cont_emitted = 0
            while remaining and cont_emitted < extra_cmd_rows:
                chunk = remaining[:cmd_width]
                remaining = remaining[cmd_width:]
                cont_emitted += 1
                if remaining and cont_emitted == extra_cmd_rows:
                    chunk = (chunk[:-1] + "…") if len(chunk) > 1 else "…"
                lines.append(_box_line(cont_indent + chunk, W))
        else:
            # Compact (tight) — single row: cmd truncated + meta inline.
            if meta_inline:
                cmd_shown = cmd_text if len(cmd_text) <= cmd_width_first else (
                    cmd_text[:cmd_width_first - 1] + "…")
                content = f"{prefix}{cmd_shown}{c(C.DIM, sep)}{meta_inline}"
            else:
                cmd = cmd_text if long else _trunc(cmd_text, cmd_width)
                content = prefix + cmd
            lines.append(_box_line(content, W))

        # Record the line range this proc occupies for click + highlight mapping.
        proc_rows.append((proc_start_line, len(lines) - 1, proc.pid))

    if hidden_count > 0:
        lines.append(_box_line(
            c(C.DIM, f"…  ({hidden_count} more hidden)"), W))

    lines.append(_box_bottom(W))
    return lines, proc_rows


_USER_COLORS: dict[str, str] = {}
_COLOR_POOL = [C.CYAN, C.GREEN, C.MAGENTA, C.BLUE, C.YELLOW]


def _user_color(user: str) -> str:
    if user not in _USER_COLORS:
        _USER_COLORS[user] = _COLOR_POOL[len(_USER_COLORS) % len(_COLOR_POOL)]
    return _USER_COLORS[user]


# ── Main display ─────────────────────────────────────────────────────────────

def _term_height() -> int:
    try:
        return shutil.get_terminal_size().lines
    except Exception:
        return 24


# ── Footer hint ──────────────────────────────────────────────────────────────

# Inline keyboard-shortcut chip: dim background + bold key text. Distinct
# from `_btn` (which marks mouse-click targets) so the two interaction
# modes read differently.
_KBD_BG_RAW = "\033[48;2;55;65;81m\033[38;2;255;255;255m\033[1m"


def _kbd(label: str) -> str:
    if _COLOR_ON:
        return f"{_KBD_BG_RAW} {label} {C.RESET}"
    return f"[{label}]"


def _hint_variants(state: UIState) -> list[str]:
    """Build hint variants tailored to the current UI state. Esc is moved
    to the front whenever the user is in a non-default state so the way
    "back" is the most prominent option."""
    sep = c(C.DIM, " · ")

    needs_reset = (
        state.mode != "gpu" or state.focus_procs
        or state.long or state.filter_gpu is not None
    )

    # Esc gets a contextual label so users see exactly what it'll undo.
    if state.focus_procs and state.mode == "gpu":
        esc_label = "exit focus"
    elif state.mode != "gpu" and not state.focus_procs and not state.long:
        esc_label = "back to gpu"
    else:
        esc_label = "reset"
    esc_part = f"{_kbd('Esc')} {c(C.DIM, esc_label)}"

    long_long  = "short cmd" if state.long else "long cmd"
    long_short = "short" if state.long else "long"

    # Mode-switch chips are only useful from the default gpu view (in
    # cpu/mem the user is already in one of the modes; in focus they're
    # focused on procs and don't need to switch).
    show_switches = state.mode == "gpu" and not state.focus_procs

    parts_long: list[str] = []
    parts_short: list[str] = []
    parts_min: list[str] = []
    if needs_reset:
        parts_long.append(esc_part)
        parts_short.append(esc_part)
        parts_min.append(esc_part)
    if show_switches:
        parts_long += [
            f"{_kbd('c')} {c(C.DIM, 'sort by CPU')}",
            f"{_kbd('m')} {c(C.DIM, 'sort by mem')}",
            f"{_kbd('p')} {c(C.DIM, 'focus procs')}",
        ]
        parts_short += [
            f"{_kbd('c')} {c(C.DIM, 'cpu')}",
            f"{_kbd('m')} {c(C.DIM, 'mem')}",
            f"{_kbd('p')} {c(C.DIM, 'focus')}",
        ]
    parts_long  += [f"{_kbd('↑↓')} {c(C.DIM, 'select')}",
                    f"{_kbd('l')} {c(C.DIM, long_long)}",
                    f"{_kbd('q')} {c(C.DIM, 'quit')}"]
    parts_short += [f"{_kbd('↑↓')} {c(C.DIM, 'sel')}",
                    f"{_kbd('l')} {c(C.DIM, long_short)}",
                    f"{_kbd('q')} {c(C.DIM, 'quit')}"]
    parts_min   += [f"{_kbd('q')} {c(C.DIM, 'quit')}"]

    return [
        " " + sep.join(parts_long),
        " " + sep.join(parts_short),
        " " + sep.join(parts_min),
        f" {_kbd('q')} {c(C.DIM, 'quit')}",
    ]


def _render_hint_line(W: int, state: UIState) -> str:
    variants = _hint_variants(state)
    for v in variants:
        if _vlen(v) <= W:
            return v
    return variants[-1]


def _hint_min_visible_len() -> int:
    """Visible width of the smallest possible hint line — used to decide
    whether the hint can fit at all in the adaptive layout."""
    return _vlen(f" {_kbd('q')} {c(C.DIM, 'quit')}")


def render_all(state: UIState | None = None) -> tuple[str, list[ClickRegion]]:
    if state is None:
        state = UIState()
    # NOTE: container_cache is persistent across ticks; `docker inspect` is
    # ~50 ms per container so we must never re-run it on every tick. Container
    # metadata (name, image, working_dir, merged_dir) does not change for a
    # container's lifetime — PID-based invalidation happens implicitly through
    # _cleanup_proc_cache → dropping per-PID entries. (Old code cleared this
    # every tick, which was the #1 avoidable hot-path call on containerised
    # hosts.)
    W = _term_width()
    H = _term_height()

    gpus = query_gpus()
    raw_procs = query_processes()
    pids = [pid for pid, _, _ in raw_procs]

    _prime_and_sample(pids)
    sysinfo = query_system()

    # query_proc_metrics (psutil cpu_percent + rss per GPU proc) is only
    # rendered in `cpu`/`mem` process-table modes; skip it in the default
    # `gpu` mode. On a host with many GPU procs this is the single biggest
    # tick-cost win because cpu_percent() reads /proc/<pid>/stat for every
    # one of them.
    if state.mode in ("cpu", "mem"):
        proc_metrics = query_proc_metrics(pids)
    else:
        proc_metrics = {}
    _cleanup_proc_cache(set(pids))

    uuid_to_gpu = {g.uuid: g for g in gpus}

    procs: list[ProcessInfo] = []
    for pid, gpu_mem, gpu_uuid in raw_procs:
        gpu = uuid_to_gpu.get(gpu_uuid)
        gpu_idx = gpu.index if gpu else -1
        p = get_process_info(pid, gpu_idx, gpu_mem)
        cpu_pct, rss = proc_metrics.get(pid, (0.0, 0))
        p.cpu_percent = cpu_pct
        p.rss_bytes = rss
        procs.append(p)

    procs_by_gpu: dict[int, list[ProcessInfo]] = {}
    for p in procs:
        procs_by_gpu.setdefault(p.gpu_index, []).append(p)

    regions: list[ClickRegion] = []
    lines: list[str] = []
    lines.append(f" {cb('htop-gpu')}   {c(C.DIM, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")

    # Adaptive layout — priorities when the terminal is short:
    #   1. GPU util + VRAM (gpus box)   ← MUST show
    #   2. CPU + Memory                 ← MUST show (compact if needed)
    #   3. Processes                    ← drop first when space is tight
    n_gpus = len(gpus)
    # When W < 100 the boxed cpu/mem stacks vertically (2 × 4 rows = 8)
    # instead of rendering side by side (4 rows). Compact form is always 2.
    stacked_cpumem = W < 80

    def _cpumem_rows(compact: bool) -> int:
        if compact:
            return 2
        return 6 if stacked_cpumem else 3

    def _gpus_rows(with_sep: bool) -> int:
        if n_gpus == 0:
            return 0
        return (2 * n_gpus + 1) if with_sep else (n_gpus + 2)

    # Core overhead (always present): title line only = 1 row.
    BASE_CORE = 1
    # Proc section cost (when shown): box top + header + bottom = 3 rows,
    # plus actual proc rows.
    PROC_SECTION = 3

    # Layout priorities when space is tight (highest-priority kept first):
    #   CPU + Memory  >  GPUs  >  Processes  >  Footer hint
    # i.e. the hint footer is always the first thing we drop on a small
    # terminal, then we shrink CPU/Mem, then drop the process section.
    cfg = None
    HINT_ROWS = 1
    # 1) try with full layout AND hint
    for compact in (False, True):
        fixed = BASE_CORE + _cpumem_rows(compact) + _gpus_rows(False) + PROC_SECTION + HINT_ROWS
        if fixed + 1 <= H:
            cfg = (False, compact, True, True)
            break
    # 2) try without hint, still showing procs
    if cfg is None:
        for compact in (False, True):
            fixed = BASE_CORE + _cpumem_rows(compact) + _gpus_rows(False) + PROC_SECTION
            if fixed + 1 <= H:
                cfg = (False, compact, True, False)
                break
    # 3) drop procs, keep CPU/Mem + GPUs (everything above the procs box)
    if cfg is None:
        for compact in (False, True):
            fixed = BASE_CORE + _cpumem_rows(compact) + _gpus_rows(False)
            if fixed <= H:
                cfg = (False, compact, False, False)
                break
    if cfg is None:
        cfg = (False, True, False, False)
    gpu_separators, compact_cpumem, show_procs, show_hint = cfg

    # Focused-procs mode: skip CPU/Mem + GPUs so the process table owns the
    # entire terminal. Reset show_procs to True since we definitely want procs.
    if state.focus_procs:
        show_procs = True

    # Render CPU + memory (unless focused on procs)
    if state.focus_procs:
        pass
    elif compact_cpumem:
        cpu_row = len(lines) + 1
        lines.extend(render_cpu_mem_compact(sysinfo, W))
        regions.append(ClickRegion(cpu_row, cpu_row, 1, W, "mode:cpu"))
        regions.append(ClickRegion(cpu_row + 1, cpu_row + 1, 1, W, "mode:mem"))
    else:
        cpu_mem_row_start = len(lines) + 1
        if W >= 80:
            W_left = W // 2
            W_right = W - W_left
            lines.extend(_hstack(
                render_cpu_box(sysinfo, W_left),
                render_mem_box(sysinfo, W_right),
                W_left, W_right,
            ))
            cpu_mem_row_end = len(lines)
            regions.append(ClickRegion(
                cpu_mem_row_start, cpu_mem_row_end, 1, W_left, "mode:cpu"))
            regions.append(ClickRegion(
                cpu_mem_row_start, cpu_mem_row_end, W_left + 1, W, "mode:mem"))
        else:
            cpu_start = len(lines) + 1
            lines.extend(render_cpu_box(sysinfo, W))
            regions.append(ClickRegion(cpu_start, len(lines), 1, W, "mode:cpu"))
            mem_start = len(lines) + 1
            lines.extend(render_mem_box(sysinfo, W))
            regions.append(ClickRegion(mem_start, len(lines), 1, W, "mode:mem"))

    # GPUs — combined box, clickable to reset mode (skip when focused on procs)
    if gpus and not state.focus_procs:
        gpus_start = len(lines) + 1
        busy_gpus = {idx for idx in procs_by_gpu.keys() if idx >= 0}
        lines.extend(render_gpus_box(
            gpus, W, state=state, show_separators=gpu_separators,
            busy_gpus=busy_gpus))
        gpus_end = len(lines)
        regions.append(ClickRegion(gpus_start, gpus_end, 1, W, "mode:gpu"))

    # Proc section only renders if the adaptive tier picked show_procs=True.
    # When skipped (short terminal), CPU/Mem + GPUs get all the space.
    if show_procs:
        # Reserve a row for either the per-selection hint (when a proc is
        # selected) or the general footer hint (when not, and tier picked it).
        sel_hint = 1 if state.selected_pid is not None else 0
        general_hint = 1 if (show_hint and state.selected_pid is None) else 0
        hint_rows = max(sel_hint, general_hint)
        if state.focus_procs:
            used = BASE_CORE + PROC_SECTION + hint_rows
        else:
            gpu_rows = _gpus_rows(gpu_separators)
            used = BASE_CORE + _cpumem_rows(compact_cpumem) + gpu_rows + PROC_SECTION + hint_rows
        max_proc_rows = max(1, H - used)
        # Compact mode (default) = 1 row per proc; long mode reserves ~2 rows
        # (cmd wrap + meta on own row).
        if state.long:
            max_procs = max(1, max_proc_rows // 2)
        else:
            max_procs = max(1, max_proc_rows)

        # Build the full pre-trim list first so we can report how many were
        # hidden by the dynamic row limit.
        query_limit = max(max_procs * 2, 256)
        if state.mode == "cpu":
            full = query_top_procs("cpu_percent", limit=query_limit)
        elif state.mode == "mem":
            full = query_top_procs("rss_bytes", limit=query_limit)
        else:
            full = procs
            if state.filter_gpu is not None:
                full = [p for p in full if p.gpu_index == state.filter_gpu]
            full = sort_procs(full, state.sort_key, state.sort_desc)
        # Reserve 1 row for the "N more hidden" indicator when truncating —
        # but only if there's actually room. With max_procs == 1 we'd
        # overflow by including the indicator, so we suppress it.
        if len(full) > max_procs:
            if max_procs >= 2:
                visible = full[:max_procs - 1]
                hidden_count = len(full) - len(visible)
            else:
                visible = full[:max_procs]
                hidden_count = 0
        else:
            visible = full
            hidden_count = 0

        # Distribute spare vertical rows across procs as extra command-wrap
        # rows so the box fills the terminal evenly instead of leaving a
        # band of blank panel-bg below it.
        spare = max(0, max_proc_rows - len(visible) - (1 if hidden_count else 0))
        extra_cmd_rows = (spare // len(visible)) if visible and not state.long else 0

        proc_top_row = len(lines) + 1  # terminal row of box top (1-indexed)
        proc_base_line = len(lines)    # 0-indexed offset into `lines`
        proc_lines, proc_row_map = render_proc_box(
            visible, W, state=state, long=state.long,
            hidden_count=hidden_count, extra_cmd_rows=extra_cmd_rows)
        # Clicking "[processes]" title toggles focused view (hide cpu/mem/gpus
        # and dedicate the entire terminal to the process table).
        regions.append(ClickRegion(
            proc_top_row, proc_top_row, 1, W, "focus:procs"))
        lines.extend(proc_lines)

        # Per-proc click regions + selection-highlight row set.
        selected_rows: set[int] = set()
        for rel_s, rel_e, pid in proc_row_map:
            abs_s = proc_base_line + rel_s
            abs_e = proc_base_line + rel_e
            regions.append(ClickRegion(
                abs_s + 1, abs_e + 1, 1, W, f"select:{pid}"))
            if state.selected_pid == pid:
                for i in range(abs_s, abs_e + 1):
                    selected_rows.add(i)

        if visible:
            header_row = proc_top_row + 1
            for content_c1, content_c2, key in _proc_col_bounds(W - 4, mode=state.mode):
                t_c1 = content_c1 + _CONTENT_COL_OFFSET
                t_c2 = content_c2 + _CONTENT_COL_OFFSET
                regions.append(ClickRegion(
                    header_row, header_row, t_c1, t_c2, f"sort:{key}"))
    else:
        selected_rows = set()

    # Selected-proc action hint line — only when something is selected.
    if state.selected_pid is not None and show_procs:
        hint_parts = [
            f"{_kbd('k')} {c(C.DIM, 'Kill')}",
            f"{_kbd('↑↓')} {c(C.DIM, 'Move')}",
            f"{_kbd('Esc / ←')} {c(C.DIM, 'Deselect')}",
            f"{_kbd('q')} {c(C.DIM, 'Quit')}",
        ]
        hint_line = (
            f" {'  '.join(hint_parts)}   "
            f"{c(C.DIM, f'PID {state.selected_pid}')}"
        )
        lines.append(hint_line)
    elif show_hint:
        # General usage hint — adapts to current state, hiding shortcuts
        # that have no effect in this context.
        lines.append(_render_hint_line(W, state))

    # Apply btop-style dark panel bg to every row; the currently-selected
    # proc row (if any) gets a slightly brighter bg for highlight.
    if _COLOR_ON:
        lines = [
            _apply_row_bg(l, W,
                          _SELECTED_BG_RAW if i in selected_rows else _PANEL_BG_RAW)
            for i, l in enumerate(lines)
        ]

    return "\n".join(lines), regions


# ── Input parsing (mouse + keys) ─────────────────────────────────────────────

# xterm-style function-key escape sequences (subset we care about)
_ESC_KEYS = {
    b"\x1bOP":    "F1",
    b"\x1bOQ":    "F2",
    b"\x1bOR":    "F3",
    b"\x1bOS":    "F4",
    b"\x1b[15~":  "F5",
    b"\x1b[17~":  "F6",
    b"\x1b[18~":  "F7",
    b"\x1b[19~":  "F8",
    b"\x1b[20~":  "F9",
    b"\x1b[21~":  "F10",
    b"\x1b[A":    "UP",
    b"\x1b[B":    "DOWN",
    b"\x1b[C":    "RIGHT",
    b"\x1b[D":    "LEFT",
}
_MOUSE_SGR_RE = re.compile(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])")


def parse_input(buf: bytes, finalize: bool = False) -> tuple[list[tuple], bytes]:
    """Decode terminal input buffer into events. Unconsumed bytes are returned
    for reuse. Events: ('key', name) or ('click', row, col).

    `finalize=True` tells the parser there are no more bytes coming for this
    burst — a lone trailing ESC is then emitted as a real Esc keypress
    instead of being held back as a possible escape-sequence prefix."""
    events: list[tuple] = []
    while buf:
        # SGR mouse: ESC [ < B ; Cx ; Cy M|m
        if buf.startswith(b"\x1b[<"):
            m = _MOUSE_SGR_RE.match(buf)
            if m:
                btn, col, row = int(m.group(1)), int(m.group(2)), int(m.group(3))
                pressed = m.group(4) == b"M"
                buf = buf[m.end():]
                if btn == 0 and pressed:
                    events.append(("click", row, col))
                continue
            break  # incomplete mouse sequence, wait for more
        if buf.startswith(b"\x1b"):
            matched = False
            for seq, name in _ESC_KEYS.items():
                if buf.startswith(seq):
                    events.append(("key", name))
                    buf = buf[len(seq):]
                    matched = True
                    break
            if matched:
                continue
            # Lone ESC vs start-of-sequence: only decide once the caller
            # signals the burst is done (see drain logic in run_watch).
            if len(buf) == 1:
                if finalize:
                    events.append(("key", "\x1b"))
                    buf = b""
                break
            buf = buf[1:]
            continue
        events.append(("key", chr(buf[0])))
        buf = buf[1:]
    return events, buf


def _cycle_sort(state: UIState) -> None:
    keys = [k for k, *_ in _PROC_COLS]
    try:
        idx = keys.index(state.sort_key)
    except ValueError:
        idx = -1
    state.sort_key = keys[(idx + 1) % len(keys)]


def _cycle_gpu_filter(state: UIState, gpu_indices: list[int]) -> None:
    order = [None] + gpu_indices
    try:
        idx = order.index(state.filter_gpu)
    except ValueError:
        idx = 0
    state.filter_gpu = order[(idx + 1) % len(order)]


def _move_selection(state: UIState, regions: list[ClickRegion], delta: int) -> None:
    """Move the selected proc up (delta=-1) or down (+1) based on the proc
    click regions emitted in the last render."""
    pids = [int(r.action.split(":")[1]) for r in regions
            if r.action.startswith("select:")]
    if not pids:
        return
    if state.selected_pid in pids:
        idx = (pids.index(state.selected_pid) + delta) % len(pids)
    else:
        idx = 0 if delta >= 0 else len(pids) - 1
    state.selected_pid = pids[idx]


def _kill_selected(state: UIState, fd: int, old_settings) -> None:
    """Try to kill the currently-selected PID. Falls back to sudo kill when
    the current user lacks permission — sudo will prompt for a password on
    the raw terminal, so we briefly suspend the TUI and restore it after."""
    pid = state.selected_pid
    if not pid:
        return
    try:
        os.kill(pid, signal.SIGTERM)
        return
    except ProcessLookupError:
        state.selected_pid = None
        return
    except PermissionError:
        pass  # fall through to sudo

    # Suspend TUI (restore terminal) so sudo can prompt for password normally.
    import termios
    import tty
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    sys.stdout.write("\x1b[?1000l\x1b[?1006l\x1b[?25h\x1b[2J\x1b[H")
    sys.stdout.flush()
    print(f"sudo kill {pid}  (enter your password if prompted)")
    try:
        subprocess.run(["sudo", "kill", str(pid)], check=False)
    except Exception as e:
        print(f"kill failed: {e}")
    time.sleep(0.3)
    # Re-enter TUI
    tty.setcbreak(fd)
    sys.stdout.write("\x1b[?25l\x1b[?1000h\x1b[?1006h")
    sys.stdout.flush()


def _apply_action(state: UIState, action: str) -> None:
    kind, _, arg = action.partition(":")
    if kind == "gpu":
        try:
            g = int(arg)
        except ValueError:
            return
        state.mode = "gpu"
        state.filter_gpu = None if state.filter_gpu == g else g
    elif kind == "sort":
        if state.sort_key == arg:
            state.sort_desc = not state.sort_desc
        else:
            state.sort_key = arg
            state.sort_desc = True
    elif kind == "mode":
        if arg == "cpu":
            state.mode = "cpu"
            state.sort_key = "cpu_percent"
            state.sort_desc = True
            state.focus_procs = False
        elif arg == "mem":
            state.mode = "mem"
            state.sort_key = "rss_bytes"
            state.sort_desc = True
            state.focus_procs = False
        elif arg == "gpu":
            state.mode = "gpu"
            state.filter_gpu = None
            state.focus_procs = False
    elif kind == "focus":
        if arg == "procs":
            state.focus_procs = not state.focus_procs
    elif kind == "select":
        try:
            state.selected_pid = int(arg)
        except ValueError:
            pass


def run_watch(interval: float, long: bool = False):
    state = UIState(long=long)
    interactive = sys.stdin.isatty()

    if interactive:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        def cleanup():
            # disable mouse + restore terminal + reset bg
            sys.stdout.write("\033[?1000l\033[?1002l\033[?1006l")
            sys.stdout.write("\033[?25h\033[0m\033[2J\033[H")
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print(f"\n{C.RESET}" if _COLOR_ON else "")

        signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))
        tty.setcbreak(fd)
        # hide cursor + enable SGR mouse click reporting
        sys.stdout.write("\033[?25l\033[?1000h\033[?1006h")
        sys.stdout.flush()
    else:
        cleanup = lambda: None
        signal.signal(signal.SIGINT, lambda *_: (print(f"\n{C.RESET}" if _COLOR_ON else ""), sys.exit(0)))

    buf = b""
    try:
        while True:
            output, regions = render_all(state=state)
            # btop-style: set the dark panel bg before clearing so the whole
            # terminal (including any rows below our content) takes the dark
            # background for a cohesive dashboard look.
            if _COLOR_ON:
                print(f"{_PANEL_BG_RAW}\033[2J\033[H{output}",
                      end="", flush=True)
            else:
                print(f"\033[2J\033[H{output}", end="", flush=True)

            if not interactive:
                time.sleep(interval)
                continue

            ready, _, _ = select.select([sys.stdin], [], [], interval)
            if not ready:
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                continue
            buf += chunk
            # Drain follow-up bytes that belong to the same input burst (an
            # escape sequence may be split across multiple TTY reads). After
            # the drain quiesces, finalize=True tells the parser any lingering
            # \x1b is a real Esc, not an in-progress sequence prefix.
            while True:
                more_ready, _, _ = select.select([sys.stdin], [], [], 0.005)
                if not more_ready:
                    break
                more = os.read(fd, 4096)
                if not more:
                    break
                buf += more
            events, buf = parse_input(buf, finalize=True)

            quit_now = False
            for ev in events:
                if ev[0] == "click":
                    _, row, col = ev
                    for r in regions:
                        if (r.row_min <= row <= r.row_max
                                and r.col_min <= col <= r.col_max):
                            _apply_action(state, r.action)
                            break
                elif ev[0] == "key":
                    key = ev[1]
                    if key in ("q", "\x03", "F10"):
                        quit_now = True
                        break
                    if key in ("l", "\x0f"):
                        state.long = not state.long
                    elif key == "a":
                        state.filter_gpu = None
                    elif key.isdigit() and len(key) == 1:
                        g = int(key)
                        state.filter_gpu = None if state.filter_gpu == g else g
                    # Mode-switch shortcuts. `g` is intentionally omitted —
                    # Esc already returns to the default gpu view.
                    elif key == "c":
                        _apply_action(state, "mode:cpu")
                    elif key == "m":
                        _apply_action(state, "mode:mem")
                    elif key == "p":
                        _apply_action(state, "focus:procs")
                    elif key == "F5":
                        _cycle_sort(state)
                    elif key == "F6":
                        gpu_indices = sorted({int(r.action.split(":")[1])
                                               for r in regions
                                               if r.action.startswith("gpu:")})
                        _cycle_gpu_filter(state, gpu_indices)
                    elif key in ("UP", "DOWN"):
                        _move_selection(state, regions, -1 if key == "UP" else 1)
                    elif key in ("k", "F9"):
                        _kill_selected(state, fd, old_settings)
                    elif key in ("\x1b", "LEFT"):
                        # Esc / ← — universal "back": clear every transient
                        # mode (selection, focus, cpu/mem mode, GPU filter,
                        # long-cmd) and return to the default GPU view.
                        state.selected_pid = None
                        state.focus_procs = False
                        state.mode = "gpu"
                        state.filter_gpu = None
                        state.long = False
            if quit_now:
                break
    finally:
        cleanup()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Terminal dashboard for NVIDIA GPUs, CPU/memory, and processes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  htop-gpu                One-shot snapshot\n"
               "  htop-gpu -w             Interactive watch mode (refresh every 1s)\n"
               "  hgpu -w                 Short alias\n"
               "  htop-gpu -wl            Watch + long command lines\n"
               "  htop-gpu --json | jq    JSON output for scripting\n",
    )
    parser.add_argument("-l", "--long", action="store_true",
                        help="show full command lines (no truncation)")
    parser.add_argument("-w", "--watch", action="store_true",
                        help="continuous monitoring mode")
    parser.add_argument("-n", "--interval", type=float, default=1.0,
                        help="refresh interval in seconds (default: 1.0)")
    parser.add_argument("--json", action="store_true",
                        help="output as JSON")
    parser.add_argument("--demo", action="store_true",
                        help=argparse.SUPPRESS)  # internal: render fake data
    args = parser.parse_args()

    if args.demo:
        # `dev/demo.py` is a local-only file (gitignored, not shipped to
        # PyPI) used for recording the README GIF. Loaded via importlib so
        # we don't ship it inside the package.
        import importlib.util
        import pathlib
        candidates = [
            pathlib.Path(__file__).resolve().parent.parent / "dev" / "demo.py",
            pathlib.Path.cwd() / "dev" / "demo.py",
        ]
        spec = None
        for p in candidates:
            if p.is_file():
                spec = importlib.util.spec_from_file_location("_htop_gpu_demo", p)
                break
        if spec is None:
            print("--demo requires dev/demo.py (dev-only, not shipped)",
                  file=sys.stderr)
            sys.exit(1)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.install()

    if args.json:
        import json
        gpus = query_gpus()
        raw_procs = query_processes()
        pids = [pid for pid, _, _ in raw_procs]
        _prime_and_sample(pids)
        sysinfo = query_system()
        proc_metrics = query_proc_metrics(pids)
        uuid_to_gpu = {g.uuid: g for g in gpus}
        procs = []
        for pid, gpu_mem, gpu_uuid in raw_procs:
            gpu = uuid_to_gpu.get(gpu_uuid)
            gpu_idx = gpu.index if gpu else -1
            p = get_process_info(pid, gpu_idx, gpu_mem)
            cpu_pct, rss = proc_metrics.get(pid, (0.0, 0))
            procs.append({
                "pid": p.pid, "user": p.user, "gpu_index": p.gpu_index,
                "gpu_mem_mib": p.gpu_mem, "cwd": p.cwd, "cmd": p.cmd,
                "elapsed": p.elapsed, "conda_env": p.conda_env,
                "venv": p.venv,
                "container_name": p.container_name,
                "container_image": p.container_image,
                "cpu_percent": cpu_pct,
                "rss_bytes": rss,
            })
        gpu_data = []
        for g in gpus:
            gpu_data.append({
                "index": g.index, "name": g.name,
                "mem_total": g.mem_total, "mem_used": g.mem_used,
                "mem_free": g.mem_free, "gpu_util": g.gpu_util,
                "mem_util": g.mem_util, "temp": g.temp,
                "power_draw": g.power_draw, "power_limit": g.power_limit,
            })
        system_data = {
            "cpu_per_core": sysinfo.cpu_per_core,
            "cpu_total": sysinfo.cpu_total,
            "load_avg": list(sysinfo.load_avg),
            "mem_total": sysinfo.mem_total,
            "mem_used": sysinfo.mem_used,
            "mem_available": sysinfo.mem_available,
            "mem_percent": sysinfo.mem_percent,
            "swap_total": sysinfo.swap_total,
            "swap_used": sysinfo.swap_used,
            "swap_percent": sysinfo.swap_percent,
            "uptime_seconds": sysinfo.uptime_seconds,
        }
        print(json.dumps({"system": system_data, "gpus": gpu_data, "processes": procs}, indent=2))
    elif args.watch:
        run_watch(args.interval, long=args.long)
    else:
        state = UIState(long=args.long)
        output, _ = render_all(state=state)
        print(output)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_int(s: str) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


if __name__ == "__main__":
    main()
