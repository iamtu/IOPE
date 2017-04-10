# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:52:29 2015

@author: dhbk
"""
import sys, os, shutil
mypath = ['./common', 
            './ML-OPE','./New1ML-OPE','./New2ML-OPE','./New3ML-OPE','./New4ML-OPE',
          './Online-OPE','./New1Online-OPE','./New2Online-OPE','./New3Online-OPE','./New4Online-OPE',
         ]
for temp in mypath:
    sys.path.insert(0, temp)
import utilities

import run_ML_OPE
import run_New1ML_OPE
import run_New2ML_OPE
import run_New3ML_OPE
import run_New4ML_OPE


import run_Online_OPE
import run_New1Online_OPE
import run_New2Online_OPE
import run_New3Online_OPE
import run_New4Online_OPE


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
    methods = ['ml-ope', 'new1ml-ope','new2ml-ope','new3ml-ope','new4ml-ope',  
    		'online-ope','new1online-ope','new2online-ope','new3online-ope','new4online-ope' ]
    method_low = method_name.lower()    

    if method_low == 'ml-ope':        
        run_mlope = run_ML_OPE.runMLOPE(train_file, settings, model_folder, test_data, tops)
        run_mlope.run()
    elif method_low == 'new1ml-ope':        
        run_new1_mlope = run_New1ML_OPE.runNew1MLOPE(train_file, settings, model_folder, test_data, tops)
        run_new1_mlope.run()
    elif method_low == 'new2ml-ope':
        run_new2_mlope = run_New2ML_OPE.runNew2MLOPE(train_file, settings, model_folder, test_data, tops)
        run_new2_mlope.run()
    elif method_low == 'new3ml-ope':
        run_new3_mlope = run_New3ML_OPE.runNew3MLOPE(train_file, settings, model_folder, test_data, tops)
        run_new3_mlope.run()
    elif method_low == 'new4ml-ope':
        run_new4_mlope = run_New4ML_OPE.runNew4MLOPE(train_file, settings, model_folder, test_data, tops)
        run_new4_mlope.run()
            

    elif method_low == 'online-ope':
        run_onlineope = run_Online_OPE.runOnlineOPE(train_file, settings, model_folder, test_data, tops)
        run_onlineope.run()
    elif method_low == 'new1online-ope':
        run_new1_onlineope = run_New1Online_OPE.runNew1OnlineOPE(train_file, settings, model_folder, test_data, tops)
        run_new1_onlineope.run()
    elif method_low == 'new2online-ope':
        run_new2onlineope = run_New2Online_OPE.runNew2OnlineOPE(train_file, settings, model_folder, test_data, tops)
        run_new2onlineope.run()
    elif method_low == 'new3online-ope':
        run_new3onlineope = run_New3Online_OPE.runNew3OnlineOPE(train_file, settings, model_folder, test_data, tops)
        run_new3onlineope.run()
    elif method_low == 'new4online-ope':
        run_new4onlineope = run_New4Online_OPE.runNew4OnlineOPE(train_file, settings, model_folder, test_data, tops)
        run_new4onlineope.run()
    elif method_low == 'streaming-ope':
        run_streamingope = run_Streaming_OPE.runStreamingOPE(train_file, settings, model_folder, test_data, tops)
        run_streamingope.run()
    
    else:
        print '\ninput wrong method name: %s\n'%(method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s'%(method)
        exit()
        
if __name__ == '__main__':
    main()
