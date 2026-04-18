# Changelog

## 0.1.7 — 2026-04-18

Performance: ~5x lighter per tick on multi-GPU hosts. Identical UI.

Measured on 8× A100 (host with live training jobs), 1 s refresh:

    median 27.95 ms/tick → 5.34 ms/tick      (5.2x)

Per-process detail (cmd, cwd, conda env, venv, docker container,
tmux/screen ancestry) is now cached for a PID's lifetime instead of
being recomputed from /proc every tick — the previous hot path opened
4–5 `/proc/<pid>/*` files and walked up the parent tree for every GPU
process on every tick.  Static GPU data (UUID, model, total memory,
enforced power limit) is likewise fetched once per handle.  `psutil`
now reads CPU total and boot time cheaply (mean-only and cache-once,
respectively); the docker-inspect cache persists across ticks instead
of being cleared every frame; `query_proc_metrics` (psutil cpu_percent
+ rss) is skipped in the default GPU process-table mode, where that
data is not rendered.  The gradient-color table is precomputed as a
LUT so bar rendering is indexed, not interpolated.

Slow-moving telemetry is re-sampled at a slightly lower cadence
behind the scenes (fan / temp / power every ~3 s, and the NVML
compute-proc enumeration — a ~1 ms × N-GPUs driver round-trip, the
single biggest per-tick cost — every ~3.5 s).  User-visible GPU util
and memory are fresh on every tick.

## 0.1.6 — 2026-04-18

- Bars now have subtle `▕ ▏` track markers on both ends, so the
  right-hand limit of the bar reads at a glance even at 100% fill.
- Tighter spacing between bars and their value/percent labels.

## 0.1.5 — 2026-04-18

- GPU row label is now colored by activity: green when no process is
  running on that GPU, red when it's busy (matches the bar gradient's
  endpoints).
- CPU box subtitle now shows core count and model name
  (e.g. `32% · 20 cores · 12th Gen Intel Core i7-12700K`), dropping the
  model then the core count as the terminal narrows.
- Memory box subtitle now shows swap in GB
  (e.g. `35% · swap 3.1G / 4.7G (66%)`), also with graceful shrinking.
- Brighter subtitle rendering on the CPU / memory / GPU boxes — the
  dim styling made model names hard to read.
- Cleaner separators: single space around `·`, around `/` in
  VRAM (`5.9G / 47.8G`) and Power (`194 / 300W`).
- GPU row now drops whole columns (Fan → Temp → Power) when the
  terminal is narrow, instead of mid-column `Fan…` truncation.

## 0.1.4 — 2026-04-18

CI: collapse build + publish into a single job — eliminates the
`upload-artifact` / `download-artifact` steps that still pulled in
Node 20. No user-visible changes.

## 0.1.3 — 2026-04-18

CI: force `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` so the few actions
still pinned to Node 20 (upload-artifact, download-artifact) silently
run under Node 24. No user-visible changes.

## 0.1.2 — 2026-04-18

CI: bump action versions to ones running on Node 24 (silences the
runner deprecation warnings). No user-visible changes.

## 0.1.1 — 2026-04-18

Fix: Power column alignment when a card draws more than 999 W (B200,
H200). Smoke-tests the GitHub Actions trusted-publisher pipeline.

## 0.1.0 — 2026-04-18

First release. GPU + CPU/memory + processes in one terminal dashboard.
Mouse-driven (panels and column headers are clickable), full keyboard
fallback, process kill with sudo prompt when needed, JSON output for
scripts. Linux only for now.
