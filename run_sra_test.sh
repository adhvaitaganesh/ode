#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

SRR_ID=$1

echo "[STATUS] Starting isolated SRA test for: ${SRR_ID}"
echo "[STATUS] Docker environment loaded. Current directory: $(pwd)"

echo "[STATUS] Executing fasterq-dump..."
# Using 4 threads to match the HTCondor CPU request
fasterq-dump --split-files --include-technical --threads 4 --progress "${SRR_ID}"

echo "[STATUS] fasterq-dump complete. Checking generated files:"
ls -lh

# Check if any fastq files were actually created before trying to gzip
if ls *.fastq 1> /dev/null 2>&1; then
    echo "[STATUS] Gzipping the output files..."
    gzip *.fastq
    echo "[STATUS] Gzip complete."
else
    echo "[WARNING] No .fastq files were found to gzip! The extraction may have failed or output nothing."
fi

echo "[STATUS] Final directory contents:"
ls -lh

echo "[STATUS] Test finished successfully."
