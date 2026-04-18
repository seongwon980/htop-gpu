# VHS tape for the README demo. Run from the repo root:
#   ~/go/bin/vhs docs/demo.tape   →   docs/demo.gif

Output docs/demo.gif

Set Shell         bash
Set FontSize      16
Set Width         1200
Set Height        710
Set Padding       6
Set Margin        0
Set Theme         "Catppuccin Mocha"
Set Framerate     20
Set TypingSpeed   30ms

# Launch the dashboard in long mode (full command lines + meta on its
# own row) so the layout is taller and more square — better fit for a
# README image than a wide letterbox.
Hide
Type "htop-gpu --demo -wl"
Enter
Sleep 1.5s
Show

# Just watch the dashboard tick — bars, %CPU, ELAPSED, power all update
# each refresh thanks to the demo source's gentle wobble.
Sleep 8s

Hide
Type "q"
