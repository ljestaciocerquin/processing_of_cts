#!/bin/bash
#SBATCH --time=1-05:00:00                           # Time limit hrs:min:sec
#SBATCH --job-name=DenoiseCT                         # Job name
#SBATCH --qos=a6000_qos
#SBATCH --partition=a6000                          # Partition
#SBATCH --nodelist=ptolemaeus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=112G
#SBATCH --output=/projects/disentanglement_methods/outputjobs/DenoiseCT%j.log   # Standard output and error log
pwd; hostname; date


# Source bashrc, such that the shell is setup properly
source ~/.bashrc
# Activate conda environment pyenv
source /home/l.estacio/miniconda3/bin/activate pytorch

# Load cuda and cudnn (make sure versions match)
# eval `spack load --sh cuda@11.3 cudnn@8.2.0.53-11.3`

#rsync -avv --info=progress2 --ignore-existing /data/groups/beets-tan/l.estacio/reg_data /processing/l.estacio

# Run your command
python /projects/disentanglement_methods/processing_of_cts/run.py