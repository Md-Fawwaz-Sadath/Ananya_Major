#!/bin/bash
# Transfer files to Jetson Xavier NX via SCP

if [ -z "$1" ]; then
    echo "Usage: ./transfer_to_jetson.sh <jetson-ip-address>"
    echo "Example: ./transfer_to_jetson.sh 192.168.1.100"
    exit 1
fi

JETSON_IP=$1
JETSON_USER="jetson"  # Change if different

echo "Transferring files to ${JETSON_USER}@${JETSON_IP}..."

# Create directory on Jetson
ssh ${JETSON_USER}@${JETSON_IP} "mkdir -p ~/eegnet-bci"

# Transfer files
scp -r ./* ${JETSON_USER}@${JETSON_IP}:~/eegnet-bci/

echo ""
echo "Transfer complete!"
echo ""
echo "Next steps:"
echo "1. SSH to Jetson: ssh ${JETSON_USER}@${JETSON_IP}"
echo "2. Run setup: cd ~/eegnet-bci && ./jetson_setup.sh"
