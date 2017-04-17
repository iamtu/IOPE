python run.py ml-ope ../data/nyt/nyt_train.txt ./data_settings/nyt/p_0.3.txt ../models/ml-ope/nyt/p_0.3/timerun1 ../data/nyt
python run.py ml-ope ../data/nyt/nyt_train.txt ./data_settings/nyt/p_0.3.txt ../models/ml-ope/nyt/p_0.3/timerun2 ../data/nyt
python run.py ml-ope ../data/nyt/nyt_train.txt ./data_settings/nyt/p_0.3.txt ../models/ml-ope/nyt/p_0.3/timerun3 ../data/nyt
python run.py ml-ope ../data/nyt/nyt_train.txt ./data_settings/nyt/p_0.3.txt ../models/ml-ope/nyt/p_0.3/timerun4 ../data/nyt
python run.py ml-ope ../data/nyt/nyt_train.txt ./data_settings/nyt/p_0.3.txt ../models/ml-ope/nyt/p_0.3/timerun5 ../data/nyt

python NPMI_LCP.py  ../data/nyt/nyt_train.txt ../models/ml-ope/nyt/p_0.3/timerun1
python NPMI_LCP.py  ../data/nyt/nyt_train.txt ../models/ml-ope/nyt/p_0.3/timerun2
python NPMI_LCP.py  ../data/nyt/nyt_train.txt ../models/ml-ope/nyt/p_0.3/timerun3
python NPMI_LCP.py  ../data/nyt/nyt_train.txt ../models/ml-ope/nyt/p_0.3/timerun4
python NPMI_LCP.py  ../data/nyt/nyt_train.txt ../models/ml-ope/nyt/p_0.3/timerun5
