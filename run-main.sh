#!/bin/bash

# Default port is 8000, we'll use 8080
PORT=8088

# Run uvicorn with host and port specified
uvicorn main:app --host 0.0.0.0 --port $PORT --reload