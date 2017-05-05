# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:52:29 2015

@author: dhbk
"""
import sys, os, shutil
sys.path.insert(0, './')
from execute import runMLOPE, runOnlineOPE
from common import utilities

def main():
    # Check input
    if len(sys.argv) != 6:
        print"usage: python run.py [method name] [train file] [setting file] [model folder] [test data folder]"
        exit()
    # Get environment variables
    method_name = sys.argv[1]
    train_file_name = sys.argv[2]
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
    settings = utilities.read_setting(setting_file) # settings là một dictionary

    # Read data for computing perplexities
    print'read data for computing perplexities ...'
    test_data = utilities.read_data_for_perpl(test_data_folder)
    #test_data =[wordids_1,wordcts_1,wordids_2,wordct_2]

    # Check method and run algorithm
    methods = ['ml-ope', 'ml-ope1','ml-ope2','ml-ope3','ml-ope4',
    		'online-ope','online-ope1','online-ope2','online-ope3','online-ope4']
    method_name = method_name.lower()

    if method_name not in methods:
        print '\ninput wrong method name: %s\n'%(method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s'%(method)
        exit()
    elif method_name in ['ml-ope', 'ml-ope1','ml-ope2','ml-ope3','ml-ope4'] :
        runOPEx = runMLOPE(method_name, train_file_name, settings, model_folder, test_data, tops)
    elif method_name in ['online-ope','online-ope1','online-ope2','online-ope3','online-ope4']:
        runOPEx = runOnlineOPE(method_name, train_file_name, settings, model_folder, test_data, tops)

    runOPEx.run()

if __name__ == '__main__':
    main()
