#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/stockquantbot}"
PUBLIC_HOST="${PUBLIC_HOST:-8.129.235.203}"
APP_USER="${APP_USER:-}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root: sudo bash deploy/deploy_ubuntu.sh"
  exit 1
fi

if [[ ! -d "${APP_DIR}" ]]; then
  echo "APP_DIR not found: ${APP_DIR}"
  echo "Set APP_DIR to your repo path, e.g. APP_DIR=/home/ubuntu/stockquantbot"
  exit 1
fi

if [[ -z "${APP_USER}" ]]; then
  APP_USER="$(stat -c '%U' "${APP_DIR}")"
fi

if ! id "${APP_USER}" >/dev/null 2>&1; then
  echo "APP_USER does not exist: ${APP_USER}"
  exit 1
fi

echo "[1/7] Install base packages..."
apt-get update
apt-get install -y python3 python3-venv python3-pip nginx nodejs npm

echo "[2/7] Check Node.js version..."
NODE_MAJOR="$(node -v | sed -E 's/^v([0-9]+).*/\1/')"
if [[ "${NODE_MAJOR}" -lt 18 ]]; then
  echo "Node.js >= 18 is required (current: $(node -v))."
  echo "Install Node 18+ first, then rerun this script."
  exit 1
fi

echo "[3/7] Install Python dependencies..."
runuser -u "${APP_USER}" -- bash -lc "cd '${APP_DIR}' && python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"

echo "[4/7] Build frontend..."
runuser -u "${APP_USER}" -- bash -lc "cd '${APP_DIR}/web' && npm ci && npm run build"

echo "[5/7] Install systemd service..."
SYSTEMD_SRC="${APP_DIR}/deploy/systemd/stockquantbot.service.tpl"
SYSTEMD_DST="/etc/systemd/system/stockquantbot.service"
sed -e "s#__APP_DIR__#${APP_DIR}#g" -e "s#__APP_USER__#${APP_USER}#g" "${SYSTEMD_SRC}" > "${SYSTEMD_DST}"
chmod 644 "${SYSTEMD_DST}"

echo "[6/7] Install nginx config..."
NGINX_SRC="${APP_DIR}/deploy/nginx/stockquantbot.conf.tpl"
NGINX_AVAIL="/etc/nginx/sites-available/stockquantbot.conf"
NGINX_ENABLED="/etc/nginx/sites-enabled/stockquantbot.conf"
sed -e "s#__PUBLIC_HOST__#${PUBLIC_HOST}#g" "${NGINX_SRC}" > "${NGINX_AVAIL}"
ln -sfn "${NGINX_AVAIL}" "${NGINX_ENABLED}"
nginx -t

echo "[7/7] Start services..."
systemctl daemon-reload
systemctl enable --now stockquantbot
systemctl enable --now nginx
systemctl restart stockquantbot
systemctl reload nginx

echo "Deployment complete."
echo "Health check:  http://${PUBLIC_HOST}/api/health"
echo "Web page:      http://${PUBLIC_HOST}/"
