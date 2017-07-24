# -*- coding: utf-8 -*-

import sys, os, shutil
sys.path.insert(0, './')
from execute import runOPE, runOnlineVB, runOnlineCVB0, runOnlineCGS
from common import utilities, NPMI_LCP_calculator

def main():

    if len(sys.argv) != 7:
        print"usage: python run.py [method name] [train file] [setting file] [output folder] [test data folder] [top_words each topic]"
        exit()

    method_name = sys.argv[1]
    train_file_name = sys.argv[2]
    setting_file = sys.argv[3]
    output_folder = sys.argv[4]
    test_data_folder = sys.argv[5]
    tops = int(sys.argv[6])

    # Create model folder if it doesn't exist
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    print 'reading setting ...'
    settings = utilities.read_setting(setting_file) # settings là một dictionary

    print 'read data for computing perplexities ...'
    test_data = utilities.read_data_for_perpl(test_data_folder)
    #test_data =[wordids_1,wordcts_1,wordids_2,wordct_2]

    # Check method and run algorithm
    methods = ['ml-ope', 'ml-ope1','ml-ope2','ml-ope3','ml-ope4',
    		'online-ope','online-ope1','online-ope2','online-ope3','online-ope4',
            'online-vb','online-cvb0','online-cgs']
    method_name = method_name.lower()

    if method_name not in methods:
        print '\ninput wrong method name: %s\n'%(method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s'%(method)
        exit()
    elif method_name == 'online-cgs' :
        runonlinecgs = runOnlineCGS(train_file_name, settings, output_folder, test_data, tops)
        runonlinecgs.run()

    elif method_name == 'online-cvb0' :
        runonlinecvb0 = runOnlineCVB0(train_file_name, settings, output_folder, test_data, tops)
        runonlinecvb0.run()
    elif method_name == 'online-vb' :
        runonlinevb = runOnlineVB(train_file_name, settings, output_folder, test_data, tops)
        runonlinevb.run()
    else:
        runOPEx = runOPE(method_name, train_file_name, settings, output_folder, test_data, tops)
        runOPEx.run()

    NPMI_LCP_calculator(train_file_name, output_folder, settings, tops)

if __name__ == '__main__':
    main()
