#!/bin/bash

echo "Starting scheduled training..."

while true
do
  python train.py
  echo "Sleeping for 5 minutes..."
  sleep 300
done