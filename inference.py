
import sys, os, shutil
import numpy as np
import matplotlib.pyplot as plt
import random
sys.path.insert(0, './')
from common import utilities
from simulation.lda import LDA
from scipy import spatial
def main():
    print 'enter main'
    if len(sys.argv) != 4:
        print"usage: python inference.py [setting file]  [beta file] [test data folder]"
        exit()
    
    setting_file = sys.argv[1]
    beta_file_name = sys.argv[2]
    
    print 'setting file: %s ' % setting_file
    print 'beta file name: %s ' % beta_file_name
    
    test_data_folder = sys.argv[3]
    output_folder = os.path.dirname(beta_file_name) +'/images';
    print 'output folder', output_folder
        
    # Create model folder if it doesn't exist
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    os.makedirs(output_folder + '/same_theta')
    os.makedirs(output_folder + '/not_same_theta')
 
    print 'reading setting ...'
    settings = utilities.read_setting(setting_file)

    doc_count = 1000; # number of documents in comparison

    (test_words_ids, test_counts_ids) = utilities.read_data_for_MAP(test_data_folder);
    test_doc_ids = random.sample(range(0, len(test_words_ids)), doc_count);
 
#     init LDA model with beta file name
    # FIXME -     zeta = calculate_average_doc_length(beta_file_name);
    zeta = 350
    lda_model = LDA(zeta, settings['alpha'], beta_file_name);
    
    # Check number of iterations for which OPE converge
    # 1. for real data
    THRESHOLD = 1e-3;
    iter_doc_counts = [];

#     print "For real dataset"
#     for i in xrange(len(test_doc_ids)):
#         l = lda_model.count_OPE_iterations(test_words_ids[test_doc_ids[i]], test_counts_ids[test_doc_ids[i]], THRESHOLD)
#         print "Doc :", i, 'number of iteration to converge : ', l
#         iter_doc_counts.append(l);
#     print 'average ', np.mean(iter_doc_counts);
#     
    # 2. for simulated data
    print 'for simulated data'
    for x in xrange(0, doc_count):
        (theta_true, word_ids, count_ids) = lda_model.generate_document()
        l = lda_model.count_OPE_iterations(word_ids, count_ids, THRESHOLD)
        print "Doc :", x, 'number of iteration to converge : ', l
        iter_doc_counts.append(l);
    print 'average ', np.mean(iter_doc_counts);
    
if __name__ == '__main__':
    main()
