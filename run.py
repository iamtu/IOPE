# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:52:29 2015

@author: dhbk
"""
import sys, os, shutil
mypath = ['./common',
            './ml-ope/ML-OPE','./ml-ope/ML-OPE1','./ml-ope/ML-OPE2','./ml-ope/ML-OPE3','./ml-ope/ML-OPE4',
          './online-ope/Online-OPE','./online-ope/Online-OPE1','./online-ope/Online-OPE2','./online-ope/Online-OPE3','./online-ope/Online-OPE4',
         ]
for temp in mypath:
    sys.path.insert(0, temp)
import utilities

import run_ML_OPE
import run_ML_OPE1
import run_ML_OPE2
import run_ML_OPE3
import run_ML_OPE4


import run_Online_OPE
import run_Online_OPE1
import run_Online_OPE2
import run_Online_OPE3
import run_Online_OPE4


def main():
    # Check input
    if len(sys.argv) != 6:
        print"usage: python run.py [method name] [train file] [setting file] [model folder] [test data folder]"
        exit()
    # Get environment variables
    method_name = sys.argv[1]
    train_file = sys.argv[2]
    setting_file = sys.argv[3]
    model_folder = sys.argv[4]
    test_data_folder = sys.argv[5]

    # FIXME - move me to settings
    tops = 10#int(sys.argv[5])

    # Create model folder if it doesn't exist
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
    os.makedirs(model_folder)
    # Read settings
    print'reading setting ...'
    settings = utilities.read_setting(setting_file)
    # settings là một dictionary
    #

    # Read data for computing perplexities
    print'read data for computing perplexities ...'
    test_data = utilities.read_data_for_perpl(test_data_folder)
    '''
    test_data =[wordids_1,wordcts_1,wordids_2,wordct_2]
    '''

    # Check method and run algorithm
    methods = ['ml-ope', 'ml-ope1','ml-ope2','ml-ope3','ml-ope4',
    		'online-ope','online-ope1','online-ope2','online-ope3','online-ope4' ]
    method_lowercase = method_name.lower()

    if method_lowercase == 'ml-ope':
        run_ml_ope = run_ML_OPE.runMLOPE(train_file, settings, model_folder, test_data, tops)
        run_ml_ope.run()
    elif method_lowercase == 'ml-ope1':
        run_ml_ope1 = run_ML_OPE1.runMLOPE1(train_file, settings, model_folder, test_data, tops)
        run_ml_ope1.run()
    elif method_lowercase == 'ml-ope2':
        run_ml_ope2 = run_ML_OPE2.runMLOPE2(train_file, settings, model_folder, test_data, tops)
        run_ml_ope2.run()
    elif method_lowercase == 'ml-ope3':
        run_ml_ope3 = run_ML_OPE3.runMLOPE3(train_file, settings, model_folder, test_data, tops)
        run_ml_ope3.run()
    elif method_lowercase == 'ml-ope4':
        run_ml_ope4 = run_ML_OPE4.runMLOPE4(train_file, settings, model_folder, test_data, tops)
        run_ml_ope4.run()


    elif method_lowercase == 'online-ope':
        run_online_ope = run_Online_OPE.runOnlineOPE(train_file, settings, model_folder, test_data, tops)
        run_online_ope.run()
    elif method_lowercase == 'online-ope1':
        run_online_ope1 = run_Online_OPE1.runOnlineOPE1(train_file, settings, model_folder, test_data, tops)
        run_online_ope1.run()
    elif method_lowercase == 'online-ope2':
        run_online_ope2 = run_Online_OPE2.runOnlineOPE2(train_file, settings, model_folder, test_data, tops)
        run_online_ope2.run()
    elif method_lowercase == 'online-ope3':
        run_online_ope3 = run_Online_OPE3.runOnlineOPE3(train_file, settings, model_folder, test_data, tops)
        run_online_ope3.run()
    elif method_lowercase == 'online-ope4':
        run_online_ope4 = run_Online_OPE4.runOnlineOPE4(train_file, settings, model_folder, test_data, tops)
        run_online_ope4.run()

    else:
        print '\ninput wrong method name: %s\n'%(method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s'%(method)
        exit()

if __name__ == '__main__':
    main()
