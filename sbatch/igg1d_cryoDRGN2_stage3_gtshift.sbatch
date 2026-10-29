#!/bin/bash
#SBATCH --job-name=igg1d-drgn2-s3
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/network/lw8858/cryoem/logs/igg1d-drgn2-s3-%j.out
#SBATCH --error=/scratch/network/lw8858/cryoem/logs/igg1d-drgn2-s3-%j.err

set -eo pipefail

module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/network/lw8858/conda-envs/cryodrgn

cd /home/lw8858/cryodrgn
export PYTHONPATH=/home/lw8858/cryodrgn:${PYTHONPATH:-}

DATA=/scratch/network/lw8858/cryoem/data/IgG-1D/IgG-1D
STAGE1=/scratch/network/lw8858/cryoem/runs/igg1d_cryodrgn2_stage1_gtshift
STAGE2=/scratch/network/lw8858/cryoem/runs/igg1d_cryodrgn2_stage2_fixed_gtshift
OUT=/scratch/network/lw8858/cryoem/runs/igg1d_cryodrgn2_stage3_gtshift

# Check required files
for f in \
    "$STAGE1/pose.pkl" \
    "$STAGE2/weights.pkl" \
    "$DATA/combined_poses.pkl"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing $f" >&2
        exit 1
    fi
done

echo "Starting cryoDRGN2+r Stage 3"
echo "Model: $STAGE2/weights.pkl"
echo "Initial rotations: $STAGE1/pose.pkl"
echo "GT shifts: $DATA/combined_poses.pkl"

cryodrgn abinit_het_old \
    "$DATA/images/snr0.01/sorted_particles.128.txt" \
    --ctf "$DATA/combined_ctfs.pkl" \
    --gt-shifts "$DATA/combined_poses.pkl" \
    --load "$STAGE2/weights.pkl" \
    --load-poses "$STAGE1/pose.pkl" \
    -o "$OUT" \
    --zdim 8 \
    -n 60 \
    -b 8 \
    --lr 1e-4 \
    --norm 0.0 1400.3507080078125 \
    --window-r 0.85 \
    --enc-layers 3 \
    --enc-dim 256 \
    --dec-layers 3 \
    --dec-dim 256 \
    --encode-mode resid \
    --pe-type gaussian \
    --feat-sigma 0.5 \
    --domain hartley \
    --activation relu \
    --ps-freq 5 \
    --t-extent 0 \
    --t-ngrid 1 \
    --checkpoint 5 \
    --no-analysis

echo "Stage 3 complete"
