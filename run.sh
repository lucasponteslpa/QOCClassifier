echo '###########IRIS DATA ATHENS############'
touch iris_n2_athens.txt
python3 exp.py --dataset iris --out_file iris_n2_athens.txt --provider ibmq_athens
echo '###########HABERMANS DATA ATHENS############'
touch habermans_n2_athens.txt
python3 exp.py --dataset Habermans --split 4 --out_file habermans_n2_athens.txt --provider ibmq_athens
echo '###########SKIN DATA ATHENS############'
touch skin_n2_athens.txt
python3 exp.py --dataset skin --split 4 --out_file skin_n2_athens_2.txt --provider ibmq_athens

echo '###########IRIS DATA SANTIAGO############'
touch iris_n2_santiago.txt
python3 exp.py --dataset iris --out_file iris_n2_santiago.txt --provider ibmq_santiago
echo '###########HABERMANS DATA SANTIAGO############'
touch habermans_n2_santiago.txt
python3 exp.py --dataset Habermans --split 4 --out_file habermans_n2_santiago.txt --provider ibmq_santiago
echo '###########SKIN DATA SANTIAGO############'
touch skin_n2_santiago.txt
python3 exp.py --dataset skin --split 4 --out_file skin_n2_santiago_2.txt --provider ibmq_santiago
