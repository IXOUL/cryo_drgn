#!/bin/bash
#SBATCH --job-name=igg1d-drgn2-s2
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/network/lw8858/cryoem/logs/igg1d-drgn2-s2-%j.out
#SBATCH --error=/scratch/network/lw8858/cryoem/logs/igg1d-drgn2-s2-%j.err

set -eo pipefail

module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/network/lw8858/conda-envs/cryodrgn

cd /home/lw8858/cryodrgn
export PYTHONPATH=/home/lw8858/cryodrgn:${PYTHONPATH:-}

DATA=/scratch/network/lw8858/cryoem/data/IgG-1D/IgG-1D
OUT=/scratch/network/lw8858/cryoem/runs/igg1d_cryodrgn2_stage2_fixed_gtshift

POSES=/scratch/network/lw8858/cryoem/runs/igg1d_cryodrgn2_stage1_gtshift/fixed_poses_gtshift.pkl

if [[ ! -f "$POSES" ]]; then
    echo "ERROR: Pose file not found: $POSES" >&2
    exit 1
fi

cryodrgn train_vae \
    "$DATA/images/snr0.01/sorted_particles.128.txt" \
    --poses "$POSES" \
    --ctf "$DATA/combined_ctfs.pkl" \
    -o "$OUT" \
    --zdim 8 \
    -n 30 \
    -b 8 \
    --lr 1e-4 \
    --norm 0.0 1400.3507080078125 \
    --window-r 0.85 \
    --enc-layers 3 --enc-dim 256 \
    --dec-layers 3 --dec-dim 256 \
    --encode-mode resid \
    --pe-type gaussian \
    --feat-sigma 0.5 \
    --domain hartley \
    --activation relu \
    --checkpoint 5 \
    --no-analysis