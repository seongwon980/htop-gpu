# VHS tape for the README demo. Run from the repo root:
#   ~/go/bin/vhs docs/demo.tape   →   docs/demo.gif

Output docs/demo.gif

Set Shell         bash
Set FontSize      16
Set Width         1500
Set Height        520
Set Padding       8
Set Margin        0
Set Theme         "Catppuccin Mocha"
Set Framerate     20
Set TypingSpeed   30ms

# Launch the dashboard but hide the typing — viewer only sees the
# rendered UI, not how it was started.
Hide
Type "htop-gpu --demo -w"
Enter
Sleep 1.5s
Show

# Just watch the dashboard tick — bars, %CPU, ELAPSED, power all update
# each refresh thanks to the demo source's gentle wobble.
Sleep 8s

# Quit (also hidden so the GIF ends on the dashboard).
Hide
Type "q"
