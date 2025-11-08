#!/bin/bash
# stop_server.sh


# Stop Apache httpd if running (for Nginx-only setups)
if pgrep httpd > /dev/null; then
    echo "Stopping Apache (httpd) to free up port 80..."
    sudo systemctl stop httpd
    sudo systemctl disable httpd
fi

# Stop Nginx if running
if pgrep nginx > /dev/null; then
    echo "Stopping Nginx..."
    sudo systemctl stop nginx
    sudo systemctl disable nginx
fi

# #!/bin/bash

# # Check if any httpd process is running
# isExistApp = `pgrep httpd`

# if [[ -n "$isExistApp" ]]; then
#     service httpd stop
# fi
