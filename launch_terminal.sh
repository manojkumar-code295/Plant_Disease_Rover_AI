#!/bin/bash
# AgriRover Phone Signal Receiver Terminal Launcher
cd /home/agrirover/Desktop/chili_disease_ai || exit 1

# Activate Python virtual environment
if [ -f "ai_env/bin/activate" ]; then
    source ai_env/bin/activate
fi

# Run the dedicated Phone Signal Receiver (Monitor Only)
python /home/agrirover/Desktop/chili_disease_ai/phone_signal_receiver.py

# Drop into interactive bash shell with ai_env active if monitor is exited
echo ""
echo "=========================================================="
echo " 🌾 AgriRover Terminal Shell (ai_env Active)"
echo " Type 'python phone_signal_receiver.py' to resume monitor."
echo "=========================================================="
echo ""
exec bash -i
