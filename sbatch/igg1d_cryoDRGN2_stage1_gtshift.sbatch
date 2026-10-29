#!/bin/bash
#SBATCH --job-name=igg1d-drgn2-s1
#SBATCH --output=igg1d-drgn2-s1-%j.out
#SBATCH --error=igg1d-drgn2-s1-%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/network/lw8858/cryoem/logs/%x_%j.out
#SBATCH --error=/scratch/network/lw8858/cryoem/logs/%x_%j.err

set -eo pipefail

module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/network/lw8858/conda-envs/cryodrgn

set -u

cd /home/lw8858/cryodrgn
export PYTHONPATH=/home/lw8858/cryodrgn:${PYTHONPATH:-}

DATA=/scratch/network/lw8858/cryoem/data/IgG-1D/IgG-1D
OUT=/scratch/network/lw8858/cryoem/runs/igg1d_cryodrgn2_stage1_gtshift

# Check the installed CLI points to your GT-shift-modified cryoDRGN2 source.
python -c 'import cryodrgn.commands.abinit_het_old as m; print("Command module:", m.__file__)'
cryodrgn abinit_het_old --help | grep -q -- '--gt-shifts' || {
  echo 'ERROR: installed abinit_het_old does not support --gt-shifts' >&2
  exit 1
}

# Avoid accidentally resuming/overwriting an existing benchmark.
if [ -e "$OUT/weights.30.pkl" ] || [ -e "$OUT/weights.pkl" ]; then
  echo "ERROR: checkpoint already exists in $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

cryodrgn abinit_het_old \
  "$DATA/images/snr0.01/sorted_particles.128.txt" \
  --ctf "$DATA/combined_ctfs.pkl" \
  --gt-shifts "$DATA/combined_poses.pkl" \
  --zdim 8 \
  -n 30 \
  -b 8 \
  --lr 1e-4 \
  --enc-layers 3 --enc-dim 256 --encode-mode resid \
  --dec-layers 3 --dec-dim 256 --activation relu \
  --ps-freq 5 \
  --t-extent 0 --t-ngrid 1 \
  --checkpoint 5 \
  --no-analysis \
  -o "$OUT"

echo "Stage 1 finished. Check $OUT/weights.30.pkl and $OUT/pose.30.pkl"
