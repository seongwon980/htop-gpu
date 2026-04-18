# htop-gpu

[![PyPI version](https://img.shields.io/pypi/v/htop-gpu.svg)](https://pypi.org/project/htop-gpu/)
[![Python versions](https://img.shields.io/pypi/pyversions/htop-gpu.svg)](https://pypi.org/project/htop-gpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GPU monitor that also shows the system's CPU, memory, and processes — in
one terminal dashboard. Like nvtop / gpustat, but clickable, with the conda
env / docker container / tmux session each process belongs to shown inline.

```
 htop-gpu   2026-04-18 13:55:42
╭──┤ ▶ cpu ├──────────────────────────┤ 9% ├──╮╭──┤ ▶ memory ├───────┤ 4% · swap 100% ├──╮
│ ■■···································   9.2% ││ Mem   ■············   43.1G / 1007.7G  4% │
╰──────────────────────────────────────────────╯╰──────────────────────────────────────────╯
╭──┤ ▶ gpus ├───────────────────────────────────────────────────┤ 8 × A100-SXM4-80GB ├──╮
│ GPU 0 │ Util: ■■■■■■■■······   67% │ VRAM: ■■■···   19.1G/80.0G  │ Power: 131/400W │ 48°C │
│ GPU 1 │ Util: ■■■■■■■■■■■■■  100% │ VRAM: ■■■■··   21.1G/80.0G  │ Power: 401/400W │ 48°C │
│ GPU 2 │ Util: ■■■■■■■■······   60% │ VRAM: ■■■■··   22.1G/80.0G  │ Power: 144/400W │ 33°C │
│ GPU 3 │ Util: ■■■■■■■■■■■■■  100% │ VRAM: ■■■■■■   60.0G/80.0G  │ Power: 271/400W │ 57°C │
│ GPU 4 │ Util: ··············    0% │ VRAM: ······     15M/80.0G  │ Power:  58/400W │ 26°C │
╰────────────────────────────────────────────────────────────────────────────────────────╯
╭──┤ ▶ processes ├─────────────────────────────────────────┤ 5 procs ├──╮
│ GPU      PID  USER     GPU MEM     ELAPSED  COMMAND                  │
│   0  1105384  user   19,540 MiB       21:23  python eval_VLM.py …  ·  ~/work [icl] │
│   1  1054968  user   21,596 MiB    03:24:03  python eval_VLM.py …  ·  ~/work [icl] │
│   2  1073580  user   22,604 MiB    01:48:45  python eval_VLM.py …  ·  ~/work [icl] │
│   3  1052082  user   35,920 MiB    03:31:02  python eval_VLM.py …  ·  ~/work [icl] │
│   3  1020423  user   25,506 MiB    06:07:01  python extract_fv.py …  ·  ~/work [icl_org] │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

## Install

```bash
pip install htop-gpu
```

```bash
htop-gpu          # one-shot snapshot
htop-gpu -w       # watch mode (refresh every 1s)
hgpu -w           # short alias
```

Needs Python 3.9+ and the NVIDIA driver. Uses NVML when available (fast),
falls back to `nvidia-smi` otherwise.

## What it does

The four titled panels (`cpu`, `memory`, `gpus`, `processes`) are
clickable.

- Click `cpu` or `memory` and the bottom table swaps to the top system
  processes by that metric.
- Click `gpus` (or hit Esc) to go back to GPU-only processes.
- Click `processes` to fullscreen the process table.
- Click any column header to sort by it.
- Click a process row to select it, then `k` to SIGTERM. If you don't own
  it, it falls back to `sudo kill` and asks for the password.

Keyboard works for everything too:

| key                      | what it does                               |
|--------------------------|--------------------------------------------|
| `c` / `m` / `p`          | switch to cpu / memory / focus-procs view  |
| `l`                      | toggle full command lines                  |
| `↑` `↓`                  | move selection                             |
| `k`, `F9`                | kill selected process                      |
| `Esc`, `←`               | back out (clears selection / mode / focus) |
| `0`–`9`                  | filter to that GPU index                   |
| `q`, `F10`, `Ctrl-C`     | quit                                       |

JSON output for scripts:

```bash
htop-gpu --json | jq
```

## Why another one?

Nothing else I tried showed me, in one place, *which conda env* / *which
docker container* a stray python process on GPU 5 belonged to. So I wrote
this. Mouse-driven mode switching and the process-kill shortcut grew out
of using it daily.

## Notes

- Linux only (uses `/proc` for cwd, cgroups, env vars).
- Mouse uses SGR mouse mode (`\x1b[?1006h`); works in iTerm2, Kitty,
  WezTerm, Ghostty, Alacritty, modern xterm/Konsole, Windows Terminal,
  and tmux with `mouse on`.

## Credits

UI borrows ideas from [htop](https://htop.dev),
[btop](https://github.com/aristocratos/btop), and
[nvtop](https://github.com/Syllo/nvtop). Independent project, not
affiliated with any of them.

## License

MIT.
