echo '###########IRIS DATA ATHENS############'
touch iris_n2_athens.txt
touch iris_n4_athens.txt
python3 exp.py --circuit QOCC --dataset iris --train True --out_file iris_n2_athens.txt --provider ibmq_athens
python3 exp.py --circuit QOCC --dataset iris --train True --num_samples 4 --out_file iris_n4_athens.txt --provider ibmq_athens
echo '###########HABERMANS DATA ATHENS############'
touch habermans_n2_athens.txt
touch habermans_n4_athens.txt
python3 exp.py --circuit QOCC --dataset Habermans --train True --split 4 --out_file habermans_n2_athens.txt --provider ibmq_athens
python3 exp.py --circuit QOCC --dataset Habermans --train True --split 4 --num_samples 4 --out_file habermans_n4_athens.txt --provider ibmq_athens
echo '###########SKIN DATA ATHENS############'
touch skin_n2_athens.txt
touch skin_n4_athens.txt
python3 exp.py --circuit QOCC --dataset skin --train True --split 4 --out_file skin_n2_athens_2.txt --provider ibmq_athens
python3 exp.py --circuit QOCC --dataset skin --train True --split 4 --num_samples 4 --out_file skin_n4_athens.txt --provider ibmq_athens

echo '###########IRIS DATA SANTIAGO############'
touch iris_n2_santiago.txt
touch iris_n4_santiago.txt
python3 exp.py --circuit QOCC --dataset iris --train True --out_file iris_n2_santiago.txt --provider ibmq_santiago
python3 exp.py --circuit QOCC --dataset iris --train True --num_samples 4 --out_file iris_n4_santiago.txt --provider ibmq_santiago
echo '###########HABERMANS DATA SANTIAGO############'
touch habermans_n2_santiago.txt
touch habermans_n4_santiago.txt
python3 exp.py --circuit QOCC --dataset Habermans --train True --split 4 --out_file habermans_n2_santiago.txt --provider ibmq_santiago
python3 exp.py --circuit QOCC --dataset Habermans --train True --split 4 --num_samples 4 --out_file habermans_n4_santiago.txt --provider ibmq_santiago
echo '###########SKIN DATA SANTIAGO############'
touch skin_n2_santiago.txt
touch skin_n4_santiago.txt
python3 exp.py --circuit QOCC --dataset skin --train True --split 4 --out_file skin_n2_santiago_2.txt --provider ibmq_santiago
python3 exp.py --circuit QOCC --dataset skin --train True --split 4 --num_samples 4 --out_file skin_n4_santiago.txt --provider ibmq_santiago
