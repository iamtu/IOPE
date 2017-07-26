1. before run onlinecgs do the following:
  go into algorithm/onlinecgs
    sudo pip install Cython
    python setup.py build_ext --inplace

2. To compare inferences methods run :
	python simulate.py ./data_settings/nyt/p_0.5.txt ../models/ml-ope/nyt/beta_final.dat ../data/nyt