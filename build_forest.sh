#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Error: Missing arguments."
  echo "Usage: bash build.sh <data_type> [run_id]"
  exit 1
fi

DATA_TYPE="$1"
RUN_ID="$2"  # Optional

LOG_DIR="assets/${DATA_TYPE}/forest"
mkdir -p "$LOG_DIR"

if [ -z "$RUN_ID" ]; then
  python -u build_forest.py --data_type="$DATA_TYPE" \
    | tee "${LOG_DIR}/${DATA_TYPE}_build_forest.log"
else
  python -u build_forest.py --data_type="$DATA_TYPE" --run_id="$RUN_ID" \
    | tee "${LOG_DIR}/${DATA_TYPE}_run${RUN_ID}_build_forest.log"
fi
