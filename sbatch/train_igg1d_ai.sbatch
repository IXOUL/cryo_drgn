#!/bin/bash
#SBATCH --job-name=igg1d-ai
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/network/lw8858/cryoem/logs/igg1d-ai-%j.out
#SBATCH --error=/scratch/network/lw8858/cryoem/logs/igg1d-ai-%j.err

set -eo pipefail

module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/network/lw8858/conda-envs/cryodrgn

set -u

cd /home/lw8858/cryodrgn
export PYTHONPATH=/home/lw8858/cryodrgn:${PYTHONPATH:-}

DATA=/scratch/network/lw8858/cryoem/data/IgG-1D/IgG-1D
OUT=/scratch/network/lw8858/cryoem/runs/abinitio/igg1d_ai_gtshift_bs32

# Verify custom cryoDRGN installation
python -c "import cryodrgn.commands.abinit as a; print(a.__file__)"
cryodrgn abinit --help | grep -- '--gt-poses'

# Log GPU memory every 10 seconds

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total \
    --format=csv -l 10 > "/scratch/network/lw8858/cryoem/logs/gpu-memory-${SLURM_JOB_ID}.csv" &
MONITOR_PID=$!

cleanup() {
    kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Training
cryodrgn abinit \
    "$DATA/images/snr0.01/sorted_particles.128.txt" \
    --ctf "$DATA/combined_ctfs.pkl" \
    --gt-poses "$DATA/combined_poses.pkl" \
    --zdim 4 \
    --n-imgs-pretrain 10000 \
    --n-imgs-pose-search 500000 \
    --epochs-sgd 25 \
    --batch-size-known-poses 32 \
    --batch-size-hps 8 \
    --batch-size-sgd 32 \
    --lr 1e-4 \
    --lr-pose-table 1e-3 \
    --lr-conf-table 1e-2 \
    --seed 42 \
    --no-analysis \
    -o "$OUT"