# Changelog

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
