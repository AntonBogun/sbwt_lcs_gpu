import os
import sys
#change path to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import random
import argparse
import numpy as np # type: ignore

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate random sequences.")
    parser.add_argument('--file', type=str, default='out.fastq', help='Output file path')
    parser.add_argument('--file_clean', type=str, default='out_clean.txt', help='Clean output file path')
    parser.add_argument('--min_length', type=int, default=1, help='Minimum length of sequences')
    parser.add_argument('--max_length', type=int, default=200, help='Maximum length of sequences')
    parser.add_argument('--num', type=int, default=100, help='Number of sequences')
    return parser.parse_args()

args = parse_arguments()

file_path = args.file
clean_file_path = args.file_clean
min_length = args.min_length
max_length = args.max_length
num_sequences = args.num

def generate_sequence(length):
    return ''.join(random.choice('ACGT') for _ in range(length))

k=31
#seed
# random.seed(42)
def complex(f,f_clean):
    for i in range(num_sequences):
        length = random.randint(min_length, max_length)
        sequence = generate_sequence(length)
        f.write(f"@sequence {i}\n")
        #split sequence into random amount of lines
        if length>=k:
            f_clean.write(f"{sequence}\n")
        if random.random()<0.5:
            f.write(f"{sequence}\n")
        else:
            num_iter=0
            max_iter = random.randint(1, 10)
            while len(sequence)>0 and num_iter<max_iter:
                line_length = random.randint(1, k)
                f.write(f"{sequence[:line_length]}\n")
                sequence = sequence[line_length:]
                num_iter+=1
            if len(sequence)>0:
                f.write(f"{sequence}\n")
        #with random chance add some amount of quality lines
        if random.random()<0.5:
            f.write(f"+quality {i}\n")
            if random.random()<0.5:
                n_quality_lines = random.randint(1, 5)
                for _ in range(n_quality_lines):
                    length2 = random.randint(1, 20)
                    f.write(f"{''.join(random.choice('!#$%^&*()_-=') for _ in range(length2))}\n")
            else:
                length2 = random.randint(1, 80)
                f.write(f"{''.join(random.choice('@!#$%>^&*()_+-=') for _ in range(length2))}\n")

def simple(f,f_clean):
    for i in range(num_sequences):
        length = random.randint(min_length, max_length)
        sequence = generate_sequence(length)
        f.write(f"@sequence {i}\n")
        f.write(f"{sequence}\n")
        if length>=k:
            f_clean.write(f"{sequence}\n")
with open(file_path, 'w') as f:
    with open(clean_file_path, 'w') as f_clean:
        complex(f,f_clean)

#./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i ./data/test.fastq -o ./out_data/test.fastq

#python ./gen_fastq/gen.py
#python ./gen_fastq/gen.py --file out2.fastq --file_clean out2_clean.txt
#python ./gen_fastq/gen.py --file out3.fastq --file_clean out3_clean.txt

#cp ./gen_fastq/out.fastq ./data/test.fastq
#cp ./gen_fastq/out2.fastq ./data/test2.fastq
#cp ./gen_fastq/out3.fastq ./data/test3.fastq

#./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i ./data/test.fastq ./data/test2.fastq ./data/test3.fastq -o ./out_data/test.fastq ./out_data/test2.fastq ./out_data/test3.fastq
#./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i ./data/test.fastq ./data/test2.fastq ./data/test3.fastq -o ./out_data/test.fastq ./out_data/test2.fastq ./out_data/test3.fastq > log.txt

#diff ./out_data/test.fastq gen_fastq/out_clean.txt
#diff ./out_data/test2.fastq gen_fastq/out2_clean.txt
#diff ./out_data/test3.fastq gen_fastq/out3_clean.txt


#python ./gen_fastq/gen.py --num 10000
#du -bh ./gen_fastq/out.fastq
#