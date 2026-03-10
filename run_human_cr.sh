#!/bin/bash
# Usage: ./run_human_cr.sh <PATH> <OUTPUT_NAME> <PREFIX>

FASTQ_DIR=$1
OUTPUT_NAME=$2
FILE_PREFIX=$3

# Define Paths
WORK_DIR="/scratch/chair_ccb/gahe00001/brain_human/results"
REF_PATH="/scratch/chair_ccb/gahe00001/brain_human/ref_data/refdata-gex-GRCh38-2024-A"

mkdir -p $WORK_DIR
cd $WORK_DIR

# Run Cell Ranger
# --id: Name of the output folder (We use your Folder Name)
# --sample: The string Cell Ranger looks for in filenames (e.g., Epilepsy_13122018)
cellranger count --id=$OUTPUT_NAME \
                 --transcriptome=$REF_PATH \
                 --fastqs=$FASTQ_DIR \
                 --sample=$FILE_PREFIX \
                 --create-bam=false \
                 --localcores=16 \
                 --localmem=60
