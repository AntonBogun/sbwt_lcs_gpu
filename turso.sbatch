#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH -e err.txt
#SBATCH -o out.txt
#SBATCH  -M ukko
#SBATCH  -p gpu-oversub
#SBATCH --cpus-per-gpu=64
## SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --mem=64G

#higher gcccore breaks nvcc
# module load GCCcore/11.3.0 CUDA/12.4.0 cmake/3.24.2
module load Python/3.10.4-GCCcore-11.3.0 CUDA/12.4.0 cmake/3.24.2 #match user environment to avoid location mismatch 


echo "Start on $(date)"
echo ""
# echo "$PATH"
# for i in 1 2 4 15 30
# # for i in 1
# do
#     # Copy the original file to a temporary file
#     cp main_replace.cpp main_temp.cpp

#     # Replace @REPLACE@ with the current number
#     sed -i "s/@REPLACE@/$i/g" main_temp.cpp

#     # Compile the modified file
#     nvcc -gencode arch=compute_80,code=sm_80 -o run -x cu main_temp.cpp

#     # Run the resulting executable and redirect the output
#     ./run > "out_3g_${i}.txt"
# done


# Remove the temporary file
# rm main_temp.cpp
required_dir="$HOME/scratch/sbwt_lcs_gpu"
if [ ! -d "$required_dir" ]; then
    echo "Directory $required_dir does not exist."
    exit 1
fi
#check pwd
if [ "$PWD" != "$required_dir" ]; then
    echo "Not in the required directory"
    exit 1
fi

./release.sh || { echo 'release failed' ; exit 1;}
# echo | cpp -fopenmp -dM | grep -i open
# ./build/bin/sbwt_lcs_gpu
# ./build/bin/sbwt_lcs_gpu -sbwt ./data/index.tdbg -i a -i b -o d c
# ./build/bin/sbwt_lcs_gpu -sbwt ./data/index.tdbg -if ./data/filelist.txt -of ./data/fileout_list.txt
./build/bin/sbwt_lcs_gpu -sbwt ./data/index.tdbg -if ./data/ecoli_filelist.txt -of ./data/ecoli_fileout_list.txt





echo ""
echo "End on $(date)"