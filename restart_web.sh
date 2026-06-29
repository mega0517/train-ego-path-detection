#!/usr/bin/env bash
# Restart the TEP-Net web app: kill any instance on $PORT, relaunch via nohup.
# Enable HTTP Basic Auth by setting WEB_AUTH_PASS (username defaults to 'admin',
# override with WEB_AUTH_USER). The password is read from the environment only —
# it is never written to disk or to git.
#
#   WEB_AUTH_PASS='your-password' ./restart_web.sh
#
set -e
cd "$(dirname "$0")"
PORT="${PORT:-5000}"

# Stop any running web app (frees the port).
pkill -f 'web_app.py' 2>/dev/null || true
sleep 1

if [ -n "${WEB_AUTH_PASS:-}" ]; then
    export WEB_AUTH_PASS
    [ -n "${WEB_AUTH_USER:-}" ] && export WEB_AUTH_USER
    echo "Basic Auth will be ENABLED (user: ${WEB_AUTH_USER:-admin})."
else
    echo "WARNING: WEB_AUTH_PASS not set — the app will run WITHOUT a password."
fi

nohup ./launch_web.sh > web_app.log 2>&1 &
echo "Restarted (launcher pid $!). Waiting for it to bind…"
sleep 4
tail -n 8 web_app.log
