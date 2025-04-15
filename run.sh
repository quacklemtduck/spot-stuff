#!/bin/bash

# Simple script to send a hardcoded message to a Discord webhook
python scripts/rsl_rl/train.py --task=Msc-v0 --headless
# Hardcoded webhook URL - REPLACE THIS WITH YOUR ACTUAL WEBHOOK URL
WEBHOOK_URL=""

# Hardcoded message
MESSAGE="I have stopped!"

# Escape special characters in the message
MESSAGE=$(echo "$MESSAGE" | sed 's/"/\\"/g')

# Construct JSON payload
JSON_PAYLOAD="{\"content\":\"$MESSAGE\"}"

# Send the webhook
echo "Sending message to Discord webhook..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" \
  -d "$JSON_PAYLOAD" \
  "$WEBHOOK_URL")

# Check response
if [ "$RESPONSE" -eq 204 ]; then
  echo "Message sent successfully!"
else
  echo "Failed to send message. HTTP status code: $RESPONSE"
  exit 1
fi
