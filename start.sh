#!/bin/bash
# start.sh

echo "Iniciando el servidor..."
uvicorn main:app --host 0.0.0.0 --port 10000
