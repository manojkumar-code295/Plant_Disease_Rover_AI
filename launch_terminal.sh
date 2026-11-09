#!/bin/bash
# AgriRover Phone Signal Receiver Terminal Launcher
cd /home/agrirover/Desktop/chili_disease_ai || exit 1

# Activate Python virtual environment
if [ -f "ai_env/bin/activate" ]; then
    source ai_env/bin/activate
fi

# Run the dedicated BTS7960 Render Cloud Controller & Hardware Verifier
python /home/agrirover/Desktop/chili_disease_ai/bts7960_render_controller.py

# Drop into interactive bash shell with ai_env active if controller is exited
echo ""
echo "=========================================================="
echo " 🌾 AgriRover Terminal Shell (ai_env Active)"
echo " Type 'python bts7960_render_controller.py' to resume."
echo "=========================================================="
echo ""
exec bash -i

