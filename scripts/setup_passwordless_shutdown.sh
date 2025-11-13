#!/bin/bash
# Setup passwordless shutdown/reboot for training scripts
# This allows auto-shutdown after training without password prompt

echo "=================================================="
echo "🔧 SETUP PASSWORDLESS SHUTDOWN"
echo "=================================================="
echo ""
echo "This will configure sudo to allow shutdown commands"
echo "without password prompt for user: $USER"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted."
    exit 1
fi

# Create sudoers file
SUDOERS_FILE="/etc/sudoers.d/allow-shutdown-$USER"

echo ""
echo "📝 Creating sudoers configuration..."
echo "File: $SUDOERS_FILE"
echo ""

# Write sudoers rule
sudo tee "$SUDOERS_FILE" > /dev/null <<EOF
# Allow $USER to shutdown/reboot without password
$USER ALL=(ALL) NOPASSWD: /sbin/shutdown, /sbin/poweroff, /sbin/reboot, /usr/bin/systemctl poweroff, /usr/bin/systemctl reboot
EOF

# Set correct permissions
sudo chmod 0440 "$SUDOERS_FILE"

# Verify syntax
if sudo visudo -c -f "$SUDOERS_FILE" &> /dev/null; then
    echo "✅ Configuration successful!"
    echo ""
    echo "You can now use these commands without password:"
    echo "  - sudo shutdown -h now"
    echo "  - sudo poweroff"
    echo "  - sudo reboot"
    echo "  - sudo systemctl poweroff"
    echo ""
    echo "Test it with:"
    echo "  sudo shutdown -c  (this won't shutdown, just cancel)"
    echo ""
else
    echo "❌ Error in sudoers configuration!"
    sudo rm -f "$SUDOERS_FILE"
    exit 1
fi

echo "=================================================="
echo "✅ SETUP COMPLETE"
echo "=================================================="
