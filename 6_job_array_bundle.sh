#!/bin/bash

# Specify the path to the config file
config=./results/frame_tie/$1_overlap_frame_tie.txt
PSR_name="$1"

# Read the number of lines, skipping the header
n_lines=$(($(wc -l < "$config") - 1))

# Dynamically extract MaxArraySize from Slurm config
MaxArraySize=$(scontrol show config | awk -F= '/MaxArraySize/ {print $2}' | tr -d ' ')

# Determine bundling strategy
if [ "$n_lines" -le "$MaxArraySize" ]; then
    tasks_per_job=1
    num_jobs=$n_lines
else
    tasks_per_job=$(( (n_lines + MaxArraySize - 1) / MaxArraySize ))
    num_jobs=$(( (n_lines + tasks_per_job - 1) / tasks_per_job ))
fi

# Generate a unique job script
timestamp=$(date +"%Y%m%d_%H%M%S")
job_script="job_script_${timestamp}.sh"

cat <<EOF > "$job_script"
#!/bin/bash -l

#SBATCH --job-name=VLBI         # Name of your job
#SBATCH --account=rc-onboard
#SBATCH --partition=interactive
#SBATCH --qos=qos_interactive
#SBATCH --time=0-12:00:00
#SBATCH --output=%x_%A_%a.out   # Output file
#SBATCH --error=%x_%A_%a.err    # Error file
#SBATCH --ntasks=1              # 1 task per job
#SBATCH --mem-per-cpu=10g       # 10GB RAM per CPU
#SBATCH --array=0-$((num_jobs - 1))  # Array size

config="${config}"
PSR_name="${PSR_name}"
tasks_per_job="${tasks_per_job}"
num_jobs="${num_jobs}"

conda init bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate VLBI

# Define the starting and ending indices for this job
start_idx=\$((SLURM_ARRAY_TASK_ID * tasks_per_job + 2))  # Start from second line
end_idx=\$((start_idx + tasks_per_job - 1))

# Debug: Check the start and end indices
echo "SLURM_ARRAY_TASK_ID = \$SLURM_ARRAY_TASK_ID"
echo "tasks_per_job = \$tasks_per_job"
echo "num_jobs = \$num_jobs"
echo "Processing lines from \$start_idx to \$end_idx"
echo " "

# Ensure we don't go past the total number of lines
if [ "\$end_idx" -gt "$n_lines" ]; then
    end_idx=$n_lines
fi

# Process lines from config file, skipping the header
while read -r ArrayTaskID RAJ DECJ PX PMRA PMDEC POSEPOCH; do
    echo "DEBUG: Processing ArrayTaskID = \$ArrayTaskID"
    echo "DEBUG: RAJ = \$RAJ, DECJ = \$DECJ, PX = \$PX, PMRA = \$PMRA, PMDEC = \$PMDEC, POSEPOCH = \$POSEPOCH"

#    output_file="output_${SLURM_ARRAY_JOB_ID}_\${ArrayTaskID}.txt"
#    echo "\${PSR_name}, \${ArrayTaskID}, RAJ = \${RAJ}, DECJ = \${DECJ}, PX = \${PX}, PMRA = \${PMRA}, PMDEC = \${PMDEC}, POSEPOCH = \${POSEPOCH}." >> "\$output_file"

    srun --mem-per-cpu=10g python3 -u 6_calculate_posterior.py "\${PSR_name}" "\${ArrayTaskID}" "\${RAJ}" "\${DECJ}" "\${PX}" "\${PMRA}" "\${PMDEC}" "\${POSEPOCH}" < /dev/null

done < <(sed -n "\${start_idx},\${end_idx}p" "$config")



EOF

# Submit the job and remove the original script after submission
sbatch "$job_script" && rm -f "$job_script"
