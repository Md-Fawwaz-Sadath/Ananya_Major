#!/bin/bash
# START HERE - Interactive deployment helper

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  EEGNet Jetson Xavier NX Deployment - INTERACTIVE HELPER        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "This script will help you deploy to your Jetson Xavier NX"
echo ""
echo "STEP 1: Do you have your Jetson Xavier NX ready?"
echo "  - Powered on and booted"
echo "  - JetPack installed"
echo "  - Connected to network (or can use USB drive)"
echo ""
read -p "Press Enter when ready..."
echo ""
echo "STEP 2: How will you transfer files?"
echo "  A) Network (SCP) - Recommended"
echo "  B) USB Drive"
echo ""
read -p "Enter A or B: " transfer_method
echo ""

if [[ "$transfer_method" == "A" ]] || [[ "$transfer_method" == "a" ]]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  OPTION A: Network Transfer (SCP)"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "First, find your Jetson's IP address:"
    echo ""
    echo "On Jetson (with monitor/keyboard connected), run:"
    echo "  hostname -I"
    echo ""
    echo "OR check your router's connected devices page"
    echo ""
    read -p "Enter your Jetson's IP address: " jetson_ip
    echo ""
    
    if [ -z "$jetson_ip" ]; then
        echo "❌ No IP address provided. Please run this script again."
        exit 1
    fi
    
    echo "Testing connection to Jetson at $jetson_ip..."
    if ping -c 2 "$jetson_ip" > /dev/null 2>&1; then
        echo "✓ Jetson is reachable!"
    else
        echo "⚠ Cannot reach $jetson_ip"
        echo "  - Verify IP address is correct"
        echo "  - Check Jetson is on same network"
        read -p "Continue anyway? (y/n): " continue_anyway
        if [[ "$continue_anyway" != "y" ]]; then
            exit 1
        fi
    fi
    
    echo ""
    read -p "Enter Jetson username (default: jetson): " jetson_user
    jetson_user=${jetson_user:-jetson}
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  TRANSFERRING FILES TO JETSON"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    PACKAGE=$(ls eegnet_jetson_deployment_*.tar.gz 2>/dev/null | head -1)
    
    if [ -z "$PACKAGE" ]; then
        echo "❌ Deployment package not found!"
        echo "   Run: ./create_deployment_package.sh"
        exit 1
    fi
    
    echo "Transferring: $PACKAGE"
    echo "To: ${jetson_user}@${jetson_ip}:~/"
    echo ""
    echo "You will be prompted for the Jetson's password..."
    echo ""
    
    scp "$PACKAGE" "${jetson_user}@${jetson_ip}:~/"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Transfer successful!"
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  NEXT: CONNECT TO JETSON AND RUN SETUP"
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        echo "1. Connect to Jetson:"
        echo "   ssh ${jetson_user}@${jetson_ip}"
        echo ""
        echo "2. Extract package:"
        echo "   cd ~"
        echo "   tar -xzf ${PACKAGE}"
        echo "   cd ${PACKAGE%.tar.gz}/"
        echo ""
        echo "3. Run setup (30-60 minutes):"
        echo "   chmod +x jetson_setup.sh"
        echo "   ./jetson_setup.sh"
        echo ""
        echo "4. Test inference:"
        echo "   python3 jetson_inference.py --model models/output.onnx --mode demo"
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        echo "📝 Copy these commands to run on Jetson:"
        echo ""
        cat > jetson_commands.txt << JETSON_EOF
# Run these commands ON THE JETSON (after SSH)
cd ~
tar -xzf ${PACKAGE}
cd ${PACKAGE%.tar.gz}/
chmod +x jetson_setup.sh
./jetson_setup.sh
# After setup completes:
python3 jetson_inference.py --model models/output.onnx --mode demo
JETSON_EOF
        cat jetson_commands.txt
        echo ""
        echo "✓ Commands saved to: jetson_commands.txt"
        echo ""
    else
        echo ""
        echo "❌ Transfer failed!"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Verify IP address: ping $jetson_ip"
        echo "  2. Check username (default is 'jetson' or 'nvidia')"
        echo "  3. Verify password"
        echo "  4. Try Option B (USB Drive) instead"
    fi
    
elif [[ "$transfer_method" == "B" ]] || [[ "$transfer_method" == "b" ]]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  OPTION B: USB Drive Transfer"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    PACKAGE=$(ls eegnet_jetson_deployment_*.tar.gz 2>/dev/null | head -1)
    
    if [ -z "$PACKAGE" ]; then
        echo "❌ Deployment package not found!"
        echo "   Run: ./create_deployment_package.sh"
        exit 1
    fi
    
    echo "Package to copy: $PACKAGE"
    echo "Size: $(du -h $PACKAGE | cut -f1)"
    echo ""
    echo "INSTRUCTIONS:"
    echo ""
    echo "1. Insert USB drive into THIS computer"
    echo ""
    read -p "Enter USB mount point (e.g., /media/usb): " usb_mount
    
    if [ -z "$usb_mount" ] || [ ! -d "$usb_mount" ]; then
        echo ""
        echo "Please mount your USB drive first:"
        echo "  sudo mount /dev/sda1 /mnt/usb"
        echo ""
        echo "Then manually copy:"
        echo "  cp $PACKAGE /mnt/usb/"
        echo "  sync"
        echo "  sudo umount /mnt/usb"
    else
        echo ""
        echo "Copying to USB..."
        cp "$PACKAGE" "$usb_mount/" && sync
        
        if [ $? -eq 0 ]; then
            echo "✓ Copied to USB!"
            echo ""
            echo "Now:"
            echo "  1. Safely eject USB from this computer"
            echo "  2. Move USB to Jetson"
            echo "  3. On Jetson, mount USB and copy file"
        fi
    fi
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ON JETSON: Run these commands"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    cat > jetson_usb_commands.txt << JETSON_EOF
# 1. Mount USB (on Jetson)
sudo mkdir -p /mnt/usb
sudo mount /dev/sda1 /mnt/usb

# 2. Copy package
cp /mnt/usb/${PACKAGE} ~/

# 3. Unmount USB
sudo umount /mnt/usb

# 4. Extract and setup
cd ~
tar -xzf ${PACKAGE}
cd ${PACKAGE%.tar.gz}/
chmod +x jetson_setup.sh
./jetson_setup.sh

# 5. After setup, test inference
python3 jetson_inference.py --model models/output.onnx --mode demo
JETSON_EOF
    cat jetson_usb_commands.txt
    echo ""
    echo "✓ Commands saved to: jetson_usb_commands.txt"
    
else
    echo "Invalid option. Please run again and choose A or B."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  📚 DOCUMENTATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Full guides available in workspace:"
echo "  • NEXT_STEPS_GUIDE.md     - Detailed step-by-step"
echo "  • DEPLOYMENT_JETSON.md    - Complete deployment guide"
echo "  • DEPLOYMENT_COMPLETE.txt - Quick reference"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Need help? Share any error messages or output with the AI assistant!"
echo ""
