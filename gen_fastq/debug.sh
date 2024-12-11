#!/bin/bash




python ./gen_fastq/gen.py --num 10000


cp ./gen_fastq/out.fastq ./data/test.fastq
cp ./gen_fastq/out2.fastq ./data/test2.fastq
cp ./gen_fastq/out3.fastq ./data/test3.fastq

./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i ./data/test.fastq ./data/test2.fastq ./data/test3.fastq -o ./out_data/test.fastq ./out_data/test2.fastq ./out_data/test3.fastq > log.txt

diff ./out_data/test.fastq gen_fastq/out_clean.txt
diff ./out_data/test2.fastq gen_fastq/out2_clean.txt
diff ./out_data/test3.fastq gen_fastq/out3_clean.txt

cp ./gen_fastq/out_break.fastq ./data/test_break.fastq
./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i ./data/test_break.fastq ./data/test2.fastq ./data/test3.fastq -o ./out_data/test_break.fastq ./out_data/test2.fastq ./out_data/test3.fastq
./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i ./data/test_break.fastq ./data/test2.fastq ./data/test3.fastq -o ./out_data/test_break.fastq ./out_data/test2.fastq ./out_data/test3.fastq > log.txt
./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i ./data/test_break.fastq -o ./out_data/test_break.fastq > log.txt
diff ./out_data/test_break.fastq gen_fastq/out_clean_break.txt
diff ./out_data/test2.fastq gen_fastq/out2_clean.txt
diff ./out_data/test3.fastq gen_fastq/out3_clean.txt


cp ./gen_fastq/out_break_2.fastq ./data/test_break_2.fastq
./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i ./data/test_break_2.fastq  -o ./out_data/test_break_2.fastq

#python ./gen_fastq/gen.py --file out2.fastq --file_clean out2_clean.txt
#cp ./gen_fastq/out2.fastq ./data/test2.fastq
#./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i ./data/test2.fastq  -o ./out_data/test2.fastq
# ./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i ./gen_fastq/test_out.fastq  -o ./out_data/test_out.fastq
#diff ./out_data/test2.fastq gen_fastq/out2_clean.txt
#diffbyline ./out_data/test_break.fastq gen_fastq/out_clean_break.txt



valgrind --tool=callgrind ./build/bin/sbwt_lcs_gpu -gpu 0 -sbwt ./data/index.tdbg -i /scratch/dongelr1/bogunant/SBWT-Search/benchmark_objects/unzipped_seqs/ERR3404625_1.fastq -o ./out_data/test_large.fastq