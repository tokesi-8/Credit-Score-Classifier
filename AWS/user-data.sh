#!/bin/bash
# Bootstrap script for an EC2 instance hosting the Streamlit app.
# Paste into EC2 launch wizard -> Advanced details -> User data.
# Edit the three variables below before launching.

set -eu

# -------- EDIT THESE -----------------------------------------------------
GIT_REPO="https://github.com/<your-handle>/<your-repo>.git"
SUBFOLDER=""                  # path inside repo; leave empty if app at root
APP_FILE="streamlit_app.py"
ENDPOINT_NAME="wine-endpoint"
# -------------------------------------------------------------------------

REGION="us-east-1"
APP_DIR="/opt/wine-app"
VENV_DIR="/opt/streamlit-venv"

if [ -z "$SUBFOLDER" ]; then
  APP_PATH="$APP_DIR"
else
  APP_PATH="$APP_DIR/$SUBFOLDER"
fi

dnf update -y
dnf install -y python3 python3-pip git

git clone "$GIT_REPO" "$APP_DIR"
chown -R ec2-user:ec2-user "$APP_DIR"

if [ ! -f "$APP_PATH/$APP_FILE" ]; then
  echo "FATAL: $APP_PATH/$APP_FILE not found."
  find "$APP_DIR" -maxdepth 4 -type f | head -40
  exit 1
fi

# Use a venv to avoid conflicts with rpm-managed system Python packages.
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install streamlit boto3

cat >/etc/systemd/system/streamlit.service <<EOF
[Unit]
Description=Streamlit App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_PATH
Environment=ENDPOINT_NAME=$ENDPOINT_NAME
Environment=AWS_REGION=$REGION
ExecStart=$VENV_DIR/bin/streamlit run $APP_FILE \\
  --server.address 0.0.0.0 \\
  --server.port 8501 \\
  --server.headless true \\
  --server.enableCORS false \\
  --server.enableXsrfProtection false
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now streamlit.service

sleep 5
if systemctl is-active --quiet streamlit; then
  touch "$APP_DIR/.userdata-success"
  chown ec2-user:ec2-user "$APP_DIR/.userdata-success"
else
  echo "FATAL: streamlit service failed to start."
  journalctl -u streamlit -n 30 --no-pager || true
  exit 1
fi
