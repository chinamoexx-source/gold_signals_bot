#!/usr/bin/env bash
# Start script for Railway/Heroku
# Expects BOT_TOKEN to be set as environment variable

if [ -z "$BOT_TOKEN" ]; then
  echo "Error: BOT_TOKEN environment variable is not set."
  echo "Set it in Railway/Heroku dashboard or export BOT_TOKEN before running."
  exit 1
fi

python telegram_gold_signals_bot.py
