#!/bin/bash

# 1. Define the manifest file
MANIFEST="samples.txt"

# 2. Check if manifest exists
if [ ! -f "$MANIFEST" ]; then
    echo "Error: $MANIFEST not found. Please run generate_list.sh first."
    exit 1
fi

# 3. Read the manifest line-by-line (skipping the header)
tail -n +2 "$MANIFEST" | while read DIR_PATH SAMPLE_NAME FILE_PREFIX; do
    
    # Skip empty lines
    if [ -z "$SAMPLE_NAME" ]; then continue; fi

    echo "Submitting Job for: $SAMPLE_NAME"
    echo "  > Path: $DIR_PATH"
    echo "  > Prefix: $FILE_PREFIX"

    # 4. Create a temporary HTCondor submit file for THIS sample
    cat <<EOT > job.sub
# --- HTCondor Submit File ---
universe = docker
docker_image = etycksen/cellranger:latest

# Resources
request_cpus = 16
request_memory = 64GB
request_disk = 50GB

# Storage Mounts (Critical for accessing /scratch)
+WantScratchMounted = true
+WantGPUHomeMounted = true

# Logging (Named after the Folder/Sample Name)
log = logs/${SAMPLE_NAME}.log
output = logs/${SAMPLE_NAME}.out
error = logs/${SAMPLE_NAME}.err

# Script to run
executable = run_human_cr.sh

# PASS THE 3 ARGUMENTS TO THE WRAPPER:
# 1. Full Path to the directory (so it knows where to look)
# 2. Sample Name (so it creates the output folder "Patient_1")
# 3. File Prefix (so it finds "Epilepsy_..._R1...")
arguments = $DIR_PATH $SAMPLE_NAME $FILE_PREFIX

queue 1
EOT

    # 5. Submit and Clean up
    condor_submit job.sub
    rm job.sub
    sleep 1

done

echo "------------------------------------------------"
echo "All jobs submitted. Monitor with 'condor_q'"
echo "------------------------------------------------"
