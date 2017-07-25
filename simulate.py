
import sys, os, shutil
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, './')
from common import utilities
from simulation.lda import LDA
from scipy import spatial
def main():
    print 'enter main'
    if len(sys.argv) != 5:
        print"usage: python simulate.py [setting file]  [beta file] [test data folder] [output folder]"
        exit()
    
    setting_file = sys.argv[1]
    beta_file_name = sys.argv[2]
    
    print 'setting file: %s ' % setting_file
    print 'beta file name: %s ' % beta_file_name
    
    test_data_folder = sys.argv[3]
    output_folder = sys.argv[4]

    # Create model folder if it doesn't exist
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    print 'reading setting ...'
    settings = utilities.read_setting(setting_file)

#     init LDA model with beta file name
    zeta = 200
    lda_model = LDA(zeta, settings['alpha'], beta_file_name)
    
#     simulating generating a document and do approximating a document
    doc_count = 10;
    for x in xrange(0, doc_count):
        (theta_true, word_ids, count_ids) = lda_model.generate_document()
        infer_generate_compare(lda_model, x, theta_true, word_ids, count_ids, output_folder);
        
#     read test documents and infer
    words_ids, counts_ids = utilities.read_data_for_MAP(test_data_folder);
    doc_count = 10;
    for x in xrange(0, doc_count):
        infer_test_doc_compare(lda_model, x, words_ids[x], counts_ids[x], output_folder);
        
def infer_test_doc_compare(lda_model, doc_id, word_ids, count_ids, output_folder):
    
    infer_iter_OPE = 50;
    infer_iter_OPE3 = 50;

    thetas_OPE = lda_model.OPE(word_ids, count_ids, infer_iter_OPE);
    map_values_OPE = lda_model.compute_MAPs(thetas_OPE, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);

    thetas_OPE3 = lda_model.OPE3(word_ids, count_ids, infer_iter_OPE3);    
    map_values_OPE3 = lda_model.compute_MAPs(thetas_OPE3, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);

    plt.figure();
    plt.plot(xrange(infer_iter_OPE), map_values_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE3), map_values_OPE3, label='OPE3');
    plt.ylabel('MAP value');
    plt.legend();
    dis_file_name = output_folder +'/' + 'doc_%d_dis.eps'%doc_id;
    plt.savefig(dis_file_name);


def infer_generate_compare(lda_model, doc_id, theta_true, word_ids, count_ids, output_folder):
    infer_iter_OPE = 50;
    infer_iter_OPE3 = 50;

    thetas_OPE = lda_model.OPE(word_ids, count_ids, infer_iter_OPE);
        
    map_values_OPE = lda_model.compute_MAPs(thetas_OPE, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    cosine_dis_OPE = compute_cosine_dis(theta_true, thetas_OPE);
    euclid_dis_OPE = compute_euclid_dis(theta_true, thetas_OPE);

    thetas_OPE3 = lda_model.OPE3(word_ids, count_ids, infer_iter_OPE3);
    
    map_values_OPE3 = lda_model.compute_MAPs(thetas_OPE3, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    cosine_dis_OPE3 = compute_cosine_dis(theta_true, thetas_OPE3);
    euclid_dis_OPE3 = compute_euclid_dis(theta_true, thetas_OPE3);
    
    
    
    plt.figure();

    plt.subplot(131)
    plt.plot(xrange(infer_iter_OPE), map_values_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE3), map_values_OPE3, label='OPE3');
    plt.ylabel('MAP value');
    plt.legend();

    plt.subplot(132)
    plt.plot(xrange(infer_iter_OPE), cosine_dis_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE3), cosine_dis_OPE3, label='OPE3');
    plt.ylabel('Cosine dis');
    plt.legend();

    plt.subplot(133)
    plt.plot(xrange(infer_iter_OPE), euclid_dis_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE3), euclid_dis_OPE3, label='OPE3');
    plt.ylabel('Euclid dis');
    plt.xlabel('iteration');
    plt.legend();
    
    dis_file_name = output_folder +'/' + 'generated_doc_%d_dis.eps'%doc_id;
    plt.savefig(dis_file_name);
    return ;

def compute_cosine_dis(theta_true, thetas):
    
    iter_count = thetas.shape[0];
    values = np.zeros(iter_count);
    for i in xrange(iter_count):
        values[i] = spatial.distance.cosine(theta_true, thetas[i]);
    return values;

def compute_euclid_dis(theta_true, thetas):
    
    iter_count = thetas.shape[0];
    values = np.zeros(iter_count);
    for i in xrange(iter_count):
        values[i] = spatial.distance.euclidean(theta_true, thetas[i]);
    return values;


if __name__ == '__main__':
    main()
