#!/bin/bash

# Jetson Xavier NX Complete Cleanup Script
# This script aggressively cleans up space on Jetson Xavier NX

echo "========================================================================"
echo "Jetson Xavier NX Complete Cleanup Script"
echo "========================================================================"
echo ""
echo "This script will remove:"
echo "  - APT cache and old packages"
echo "  - System logs"
echo "  - Python cache and bytecode"
echo "  - Temporary files"
echo "  - CUDA samples and documentation"
echo "  - Documentation and man pages"
echo "  - Thumbnail caches"
echo ""
echo "⚠️  WARNING: This is aggressive cleanup. Review before proceeding!"
echo ""
read -p "Continue with cleanup? (type 'yes' to confirm): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled"
    exit 0
fi

echo ""
echo "========================================================================"
echo "Starting Cleanup Process"
echo "========================================================================"
echo ""

# Store initial disk usage
INITIAL_USED=$(df -h / | tail -1 | awk '{print $3}')
INITIAL_AVAIL=$(df -h / | tail -1 | awk '{print $4}')

echo "Initial disk usage:"
df -h /
echo ""

# Progress counter
STEP=1
TOTAL_STEPS=15

# Function to print progress
print_step() {
    echo ""
    echo "[$STEP/$TOTAL_STEPS] $1..."
    STEP=$((STEP + 1))
}

# 1. Clean APT cache
print_step "Cleaning APT cache"
sudo apt clean
sudo apt autoclean

# 2. Remove old packages
print_step "Removing old/unused packages"
sudo apt autoremove -y

# 3. Clean pip cache
print_step "Cleaning pip cache"
pip3 cache purge 2>/dev/null || true
rm -rf ~/.cache/pip 2>/dev/null || true

# 4. Remove Python bytecode
print_step "Removing Python bytecode"
find ~/ -type f -name '*.pyc' -delete 2>/dev/null || true
find ~/ -type d -name '__pycache__' -delete 2>/dev/null || true
find ~/ -type d -name '.pytest_cache' -delete 2>/dev/null || true
find ~/ -type d -name '.ipynb_checkpoints' -delete 2>/dev/null || true

# 5. Clean system logs
print_step "Cleaning system logs"
sudo journalctl --vacuum-size=50M 2>/dev/null || true
sudo rm -rf /var/log/*.gz 2>/dev/null || true
sudo rm -rf /var/log/*.1 2>/dev/null || true
sudo rm -rf /var/log/*.old 2>/dev/null || true

# 6. Remove temporary files
print_step "Removing temporary files"
sudo rm -rf /tmp/* 2>/dev/null || true
sudo rm -rf /var/tmp/* 2>/dev/null || true

# 7. Clean thumbnail cache
print_step "Cleaning thumbnail cache"
rm -rf ~/.cache/thumbnails/* 2>/dev/null || true
rm -rf ~/.thumbnails/* 2>/dev/null || true
rm -rf ~/.cache/* 2>/dev/null || true

# 8. Remove documentation
print_step "Removing documentation (can be reinstalled if needed)"
sudo rm -rf /usr/share/doc/* 2>/dev/null || true
sudo rm -rf /usr/share/man/* 2>/dev/null || true

# 9. Clean CUDA samples
print_step "Removing CUDA samples and documentation"
sudo rm -rf /usr/local/cuda/samples 2>/dev/null || true
sudo rm -rf /usr/local/cuda/doc 2>/dev/null || true

# 10. Remove sample media
print_step "Removing sample media files"
rm -rf ~/Videos/* 2>/dev/null || true
rm -rf ~/Pictures/sample* 2>/dev/null || true
rm -rf ~/Music/sample* 2>/dev/null || true

# 11. Clean browser cache
print_step "Cleaning browser cache"
rm -rf ~/.mozilla/firefox/*/cache* 2>/dev/null || true
rm -rf ~/.cache/chromium 2>/dev/null || true

# 12. Remove old kernels (keeps current)
print_step "Removing old kernel versions"
sudo apt autoremove --purge -y 2>/dev/null || true

# 13. Clean package lists
print_step "Cleaning package lists"
sudo rm -rf /var/lib/apt/lists/* 2>/dev/null || true
sudo apt update -qq

# 14. Remove crash reports
print_step "Removing crash reports"
sudo rm -rf /var/crash/* 2>/dev/null || true

# 15. Final cleanup
print_step "Final cleanup and optimization"
sudo apt autoclean
sudo apt clean

echo ""
echo "========================================================================"
echo "Cleanup Complete!"
echo "========================================================================"
echo ""

# Show final disk usage
FINAL_USED=$(df -h / | tail -1 | awk '{print $3}')
FINAL_AVAIL=$(df -h / | tail -1 | awk '{print $4}')

echo "Disk usage after cleanup:"
df -h /
echo ""
echo "Summary:"
echo "  Before:  Used: $INITIAL_USED | Available: $INITIAL_AVAIL"
echo "  After:   Used: $FINAL_USED | Available: $FINAL_AVAIL"
echo ""

# Additional recommendations
echo "========================================================================"
echo "Additional Cleanup Options (Manual)"
echo "========================================================================"
echo ""
echo "For more space, you can manually remove:"
echo ""
echo "1. Docker (if installed):"
echo "   sudo apt purge docker-ce docker-ce-cli containerd.io -y"
echo "   sudo rm -rf /var/lib/docker"
echo ""
echo "2. Conda/Anaconda (if installed):"
echo "   rm -rf ~/miniconda3"
echo "   rm -rf ~/anaconda3"
echo ""
echo "3. Unused Python packages:"
echo "   pip3 list"
echo "   pip3 uninstall <package-name>"
echo ""
echo "4. Old datasets/projects:"
echo "   rm -rf ~/old-projects"
echo "   rm -rf ~/datasets"
echo ""
echo "5. Find largest files:"
echo "   sudo find / -type f -size +100M 2>/dev/null | xargs ls -lh"
echo ""
echo "6. Find largest directories:"
echo "   sudo du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20"
echo ""
echo "========================================================================"
echo ""
echo "✅ Cleanup script finished successfully!"
echo ""
