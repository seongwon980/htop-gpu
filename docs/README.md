# Recording the demo GIF

The README expects `docs/demo.gif` (~780px wide). Two recommended ways
to make one — pick whichever you prefer.

## Option A: VHS (declarative, repeatable, no real terminal needed)

[VHS](https://github.com/charmbracelet/vhs) lets you script a terminal
session and render to a GIF. Best when you only need keyboard input.

```bash
go install github.com/charmbracelet/vhs@latest        # or brew install vhs
vhs docs/demo.tape                                    # outputs docs/demo.gif
```

`docs/demo.tape` is included.

VHS doesn't simulate mouse clicks, so the demo focuses on the keyboard
shortcuts (`c`, `m`, `p`, `Esc`, `↑↓`, `k`, `q`).

## Option B: asciinema + agg (real terminal session, mouse works)

```bash
# install
pip install asciinema
cargo install --git https://github.com/asciinema/agg

# record (Ctrl-D to stop)
asciinema rec docs/demo.cast --cols 130 --rows 32

# inside the recording terminal:
htop-gpu -w
# … click around, press keys …
# Ctrl-D to stop

# convert to GIF
agg docs/demo.cast docs/demo.gif --font-size 14 --speed 1.2
```

This produces a real recording that captures clicks, but the GIF is
typically larger (a few MB).

## Tips

- 5–8 seconds of demo is plenty. People skim.
- Keep terminal at ~130×32 so the GIF fits in a README column.
- If the GIF gets above ~3 MB, GitHub will be slow to render — trim the
  recording or drop to fewer frames per second.
