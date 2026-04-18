# Changelog

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
