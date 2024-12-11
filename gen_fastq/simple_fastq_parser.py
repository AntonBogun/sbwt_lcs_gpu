#enforce following structure, while also reading into a clean version
#Repeat:

#@...sequence name text, irrelevant
#sequence text (only ACTG)
#+...quality name text, irrelevant
#quality text (irrelevant)

import os
import argparse


import time
# class ProgressPrinter:
# 	def __init__(self, n_jobs, total_prints):
# 		self.n_jobs = n_jobs
# 		self.processed = 0
# 		self.total_prints = total_prints
# 		self.next_print = 0
# 		self.first_print = True
# 		self.start = 0
# 	def job_done(self):
# 		if self.first_print:
# 			self.start = time.time()
# 		if self.next_print == self.processed:
# 			if not self.first_print:
# 				print('\r', end='')  # Erase current line
# 			progress_percent = int(100 * (self.processed / self.n_jobs))
# 			print(f"{progress_percent}% ({self.processed}/{self.n_jobs})", end='')
# 			# expected seconds left
# 			now = time.time()
# 			duration = now - self.start
# 			if not self.first_print:
# 				expected_duration = duration * self.n_jobs / self.processed
# 				expected_seconds_left = int(expected_duration - duration)
# 				print(f", ETA: {expected_seconds_left}s", end='', flush=True)
# 			else:
# 				print("", end='', flush=True)
# 			self.next_print += int(self.n_jobs / self.total_prints)
# 		self.processed += 1
# 		if self.processed == self.n_jobs:
# 			print(f"\r100% ({self.n_jobs}/{self.n_jobs})", flush=True)
# 		self.first_print = False
# 	def reset(self, n_jobs, total_prints):
# 		self.n_jobs = n_jobs
# 		self.processed = 0
# 		self.total_prints = total_prints
# 		self.next_print = 0
# 		self.first_print = True
#we'll do it manually instead


def parse_arguments():
    parser = argparse.ArgumentParser(description="parse fastq in a strict format.")
    parser.add_argument('--file', type=str, help='Output file path', required=True)
    parser.add_argument('--file_clean', type=str, help='Clean output file path', required=True)
    parser.add_argument('--type', type=str, default="original", help='Type of parsing')
    return parser.parse_args()

args = parse_arguments()


file_path = args.file
clean_file_path = args.file_clean



def parse(f,f_clean):

    #read file line by line
    fsize = os.path.getsize(file_path)
    print("fsize:",fsize)
    i = 0
    line_num=0
    start = time.time()
    n_prints=100
    next_print = 0
    i_prints=0
    while True:
        #read sequence name
        line = f.readline()
        if not line:
            break
        i=f.tell()
        if i>=next_print:
            now = time.time()
            duration = now - start
            expected_duration = duration * fsize / i
            print(f"{i}/{fsize} ({int(100*i/fsize)}%), {duration:.2f}s elapsed, ETA: {expected_duration - duration:.2f}s", end='\n' if (i_prints%(n_prints//10))==0 else '\r', flush=True)
            next_print += int(fsize / n_prints)
            i_prints+=1

        assert line[0] == '@', f"Expected @ at start of line {line_num} near {f.tell()}"
        line_num+=1

        #read sequence
        line = f.readline()
        if not line:
            print("didn't find sequence")
            break
        line=line.replace("N","A")
        if len(set(line.strip())-set('ACGT'))>0:
            print("found invalid characters in sequence at line",line_num)
            print(line)
            break
        f_clean.write(line)
        line_num+=1

        line = f.readline()
        if not line:
            print("didn't find quality begin")
            break
        assert line[0] == '+', f"Expected + at start of line {line_num} near {f.tell()}"
        line_num+=1

        line = f.readline()
        if not line:
            print("didn't find quality")
            break
        line_num+=1
    print("\nfinal i",i,"final line_num",line_num)
    print("done")

with open(file_path, 'r') as f:
    print(f"parsing {file_path}")
    with open(clean_file_path, 'w') as f_clean:
        print(f"writing to {clean_file_path}")
        parse(f,f_clean)
#python gen_fastq/simple_fastq_parser.py
#python gen_fastq/simple_fastq_parser.py --file /scratch/dongelr1/bogunant/SBWT-Search/benchmark_objects/unzipped_seqs/ERR3404625_1_1MB.fastq --file_clean gen_fastq/clean_parsed.fastq
#./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i /scratch/dongelr1/bogunant/SBWT-Search/benchmark_objects/unzipped_seqs/ERR3404625_1_1MB.fastq -o ./out_data/test_1MB.fastq
#diff ./out_data/test_1MB.fastq gen_fastq/clean_parsed.fastq

#python gen_fastq/simple_fastq_parser.py --file /scratch/dongelr1/bogunant/SBWT-Search/benchmark_objects/unzipped_seqs/ERR3404625_1_10MB.fastq --file_clean gen_fastq/clean_parsed_bigger.fastq
#./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i /scratch/dongelr1/bogunant/SBWT-Search/benchmark_objects/unzipped_seqs/ERR3404625_1_10MB.fastq -o ./out_data/test_10MB.fastq
#diff ./out_data/test_10MB.fastq gen_fastq/clean_parsed_bigger.fastq

#python gen_fastq/simple_fastq_parser.py --file /scratch/dongelr1/bogunant/SBWT-Search/benchmark_objects/unzipped_seqs/ERR3404625_1.fastq --file_clean gen_fastq/clean_parsed_large.fastq
#time ./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i /scratch/dongelr1/bogunant/SBWT-Search/benchmark_objects/unzipped_seqs/ERR3404625_1.fastq -o ./out_data/test_large.fastq
#diff ./out_data/test_large.fastq gen_fastq/clean_parsed_large.fastq



# /scratch/dongelr1/bogunant/datasets/ecoli/coli3682_dataset/GCA_000005845.2_ASM584v2.fna
# ll /scratch/dongelr1/bogunant/datasets/ecoli/coli3682_dataset > /scratch/dongelr1/bogunant/datasets/ecoli/LL_LIST_coli3682_dataset.txt
# ll /scratch/dongelr1/bogunant/datasets/salmonella/output > /scratch/dongelr1/bogunant/datasets/salmonella/LL_LIST_salmonella_output.txt
# cd /scratch/dongelr1/bogunant/datasets/salmonella/
# trimfile LL_LIST_salmonella_output.txt 1MB
# cd-

# /scratch/dongelr1/bogunant/datasets/salmonella/output/SAMD00011660.contigs.fa.gz