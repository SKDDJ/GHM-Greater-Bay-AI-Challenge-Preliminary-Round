#!/bin/bash

/workspace/sample.sh > /workspace/results.log 2>&1
cat /workspace/results.log | grep 'finetuned parameters' | awk '{s+=$(NF)} END {print s}'
python score.py