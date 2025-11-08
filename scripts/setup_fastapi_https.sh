#!/bin/bash

# -----------------------------------------------------------------------------
# File: scripts/setup_fastapi_https.sh
# -----------------------------------------------------------------------------
# 1) Loads environment from .env (skipping malformed lines)
# 2) Starts FastAPI (Uvicorn) via systemd on localhost:8000 (persistent)
# 3) Installs & configures Nginx & Certbot
# 4) Uses modern HTTP/2 listen directives
# 5) Adds WebSocket proxy headers
# 6) Tunes proxy buffers (no temp files)
# -----------------------------------------------------------------------------

# --- CONFIG ------------------------------------------------------------------
APP_DIR="/home/ec2-user/asr-fastapi-app"
ENV_FILE="$APP_DIR/.env"
DOMAIN="${DOMAIN:-asr-api.owusuboakye.ca}"
LE_EMAIL="${LE_EMAIL:-kwadwonyame@owusuboakye.com}"
PORT=9002
NGINX_CONF="/etc/nginx/conf.d/${DOMAIN}.conf"
FASTAPI_LOG="$APP_DIR/fastapi.log"
SERVICE_NAME="asr-fastapi"

# --- FUNCTIONS ---------------------------------------------------------------
load_env() {
  if [ -f "$1" ]; then
    grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$1" | sed 's/\r$//' > /tmp/.env_clean
    set -a; source /tmp/.env_clean; set +a
    rm /tmp/.env_clean
  fi
}

# --- START FASTAPI (systemd) -------------------------------------------------
echo "[fastapi] Loading env & creating systemd unit..."
load_env "$ENV_FILE"

# Stop old service or stray process
sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
pkill -f "uvicorn.*fastapi_app_security:app" 2>/dev/null || true

# Create/refresh unit
sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<EOF
[Unit]
Description=Asr Ocr FastAPI (Uvicorn) - proxied by Nginx
After=network.target

[Service]
User=ec2-user
WorkingDirectory=${APP_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=/bin/bash -lc 'source venv/bin/activate && uvicorn fastapi_app_security:app --host 127.0.0.1 --port ${PORT}'
Restart=always
RestartSec=5
UMask=0002
StandardOutput=append:${FASTAPI_LOG}
StandardError=append:${FASTAPI_LOG}

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"

# --- INSTALL NGINX & CERTBOT -------------------------------------------------
echo "[nginx] Installing Nginx & Certbot..."
if command -v apt >/dev/null; then
  sudo apt update
  sudo apt install -y nginx certbot python3-certbot-nginx
else
  sudo yum install -y nginx certbot python3-certbot-nginx
fi

# Stop any old web servers
sudo pkill nginx 2>/dev/null || true
for svc in nginx httpd; do
  sudo systemctl stop    "$svc" 2>/dev/null || true
  sudo systemctl disable "$svc" 2>/dev/null || true
done

# Enable & start Nginx
sudo systemctl enable --now nginx

# 1) HTTP-only block for Certbot
sudo tee "$NGINX_CONF" > /dev/null <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};

    location / {
        proxy_pass         http://127.0.0.1:${PORT};
        proxy_http_version 1.1;

        # Standard headers
        proxy_set_header   Host \$host;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
    }
}
EOF

sudo nginx -t && sudo systemctl reload nginx

# 2) Obtain/Renew Let's Encrypt cert
sudo certbot --nginx \
  --non-interactive --agree-tos --redirect \
  --email "$LE_EMAIL" \
  -d "$DOMAIN"

# 3) Final HTTPS + HTTP/2 + buffer tuning + websockets
sudo tee "$NGINX_CONF" > /dev/null <<EOF
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name ${DOMAIN};

    ssl_certificate     /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;
    include             /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam         /etc/letsencrypt/ssl-dhparams.pem;

    location / {
        proxy_pass         http://127.0.0.1:${PORT};
        proxy_http_version 1.1;

        # WebSocket support
        proxy_set_header   Upgrade \$http_upgrade;
        proxy_set_header   Connection "upgrade";

        # Increase in-memory buffers; disable disk buffering
        proxy_buffer_size        16k;
        proxy_buffers            8 16k;
        proxy_busy_buffers_size  32k;
        proxy_max_temp_file_size 0;

        # Standard headers
        proxy_set_header   Host \$host;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
    }
}

# Redirect HTTP → HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};
    return 301 https://\$host\$request_uri;
}
EOF

sudo nginx -t && sudo systemctl restart nginx

echo "[done] FastAPI is managed by systemd: sudo systemctl status ${SERVICE_NAME}"
echo "[done] Nginx proxy is live at: https://${DOMAIN}"
echo "[hint] Logs: journalctl -u ${SERVICE_NAME} -n 100 --no-pager | tail -n 100 ${FASTAPI_LOG}"
