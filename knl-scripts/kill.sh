ps aux | grep python3 | awk '{print $2}' | xargs kill -9
