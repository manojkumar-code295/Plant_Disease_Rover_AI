#!/bin/bash
# AgriRover Live Command Terminal Auto-Launcher
cd /home/agrirover/Desktop/chili_disease_ai || exit 1

# Activate Python virtual environment
if [ -f "ai_env/bin/activate" ]; then
    source ai_env/bin/activate
fi

# Run the live AgriRover monitor
python /home/agrirover/Desktop/chili_disease_ai/live_terminal_monitor.py

# Drop into interactive bash shell with ai_env active if monitor is exited
echo ""
echo "=========================================================="
echo " 🌾 AgriRover Interactive Terminal Shell (ai_env Active)"
echo " Type 'python live_terminal_monitor.py' to resume monitor."
echo "=========================================================="
echo ""
exec bash -i
