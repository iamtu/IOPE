
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
        print"usage: python simulate.py [setting file]  [beta file] [test data folder]"
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
 
    print 'reading setting ...'
    settings = utilities.read_setting(setting_file)
 
#     init LDA model with beta file name
    zeta = 200
    lda_model = LDA(zeta, settings['alpha'], beta_file_name)
     
#     simulating generating a document and do approximating a document
    doc_count = 10;
    print 'Generate %d docs to compare.'%doc_count
    for x in xrange(0, doc_count):
        (theta_true, word_ids, count_ids) = lda_model.generate_document()
        infer_generate_compare(lda_model, x, theta_true, word_ids, count_ids, output_folder);
         
#     read test documents and infer
    (words_ids, counts_ids) = utilities.read_data_for_MAP(test_data_folder);
    doc_ids = random.sample(range(0, len(words_ids)), 100);
    print 'Compare random %d docs in test data'%(len(doc_ids))
    for x in xrange(0, len(doc_ids)):
        infer_test_doc_compare(lda_model, doc_ids[x], words_ids[doc_ids[x]], counts_ids[doc_ids[x]], output_folder);
    
    
    print 'Output result in folder %s' % output_folder
        
def infer_test_doc_compare(lda_model, doc_id, word_ids, count_ids, output_folder):
    
    infer_iter_OPE = 50;
    infer_iter_OPE3 = 50;
    
    init_theta = np.random.rand(lda_model._K) + 1.
    init_theta /= sum(init_theta)
    
    thetas_OPE = lda_model.OPE(word_ids, count_ids, init_theta, infer_iter_OPE);
    map_values_OPE = lda_model.compute_MAPs(thetas_OPE, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);

    thetas_OPE3 = lda_model.OPE3(word_ids, count_ids, init_theta, infer_iter_OPE3);    
    map_values_OPE3 = lda_model.compute_MAPs(thetas_OPE3, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);

    plt.figure();
    plt.plot(xrange(infer_iter_OPE), map_values_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE3), map_values_OPE3, label='OPE3');
    plt.ylabel('MAP value');
    plt.legend();
    dis_file_name = output_folder +'/' + 'test_doc_%d_dis.eps'%doc_id;
    plt.savefig(dis_file_name);
    plt.close();


def infer_generate_compare(lda_model, doc_id, theta_true, word_ids, count_ids, output_folder):
    infer_iter_OPE = 50;
    infer_iter_OPE1 = 50;
    infer_iter_OPE2 = 50;
    infer_iter_OPE3 = 50;
    infer_iter_OPE4 = 50;
    
    
    init_theta = np.random.rand(lda_model._K) + 1.
    init_theta /= sum(init_theta)

    thetas_OPE = lda_model.OPE(word_ids, count_ids, init_theta, infer_iter_OPE);
    thetas_OPE1 = lda_model.OPE1(word_ids, count_ids, init_theta, infer_iter_OPE1);
    thetas_OPE2 = lda_model.OPE2(word_ids, count_ids, init_theta, infer_iter_OPE2);
    thetas_OPE3 = lda_model.OPE3(word_ids, count_ids, init_theta, infer_iter_OPE3);
    thetas_OPE4 = lda_model.OPE4(word_ids, count_ids, init_theta, infer_iter_OPE4, 0.01);
        
    map_values_OPE = lda_model.compute_MAPs(thetas_OPE, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    cosine_dis_OPE = compute_cosine_dis(theta_true, thetas_OPE);
    euclid_dis_OPE = compute_euclid_dis(theta_true, thetas_OPE);
    kl_divergence_OPE = compute_kl_divergences(theta_true, thetas_OPE);
    
    map_values_OPE1 = lda_model.compute_MAPs(thetas_OPE1, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    cosine_dis_OPE1 = compute_cosine_dis(theta_true, thetas_OPE1);
    euclid_dis_OPE1 = compute_euclid_dis(theta_true, thetas_OPE1);
    kl_divergence_OPE1 = compute_kl_divergences(theta_true, thetas_OPE1);

    map_values_OPE2 = lda_model.compute_MAPs(thetas_OPE2, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    cosine_dis_OPE2 = compute_cosine_dis(theta_true, thetas_OPE2);
    euclid_dis_OPE2 = compute_euclid_dis(theta_true, thetas_OPE2);
    kl_divergence_OPE2 = compute_kl_divergences(theta_true, thetas_OPE2);

    map_values_OPE3 = lda_model.compute_MAPs(thetas_OPE3, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    cosine_dis_OPE3 = compute_cosine_dis(theta_true, thetas_OPE3);
    euclid_dis_OPE3 = compute_euclid_dis(theta_true, thetas_OPE3);
    kl_divergence_OPE3 = compute_kl_divergences(theta_true, thetas_OPE3);
    
    map_values_OPE4 = lda_model.compute_MAPs(thetas_OPE4, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    cosine_dis_OPE4 = compute_cosine_dis(theta_true, thetas_OPE4);
    euclid_dis_OPE4 = compute_euclid_dis(theta_true, thetas_OPE4);
    kl_divergence_OPE4 = compute_kl_divergences(theta_true, thetas_OPE4);

    plt.figure();

    plt.subplot(221)
    plt.plot(xrange(infer_iter_OPE), map_values_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE1), map_values_OPE1, label='OPE1');
    plt.plot(xrange(infer_iter_OPE2), map_values_OPE2, label='OPE2');
    plt.plot(xrange(infer_iter_OPE3), map_values_OPE3, label='OPE3');
    plt.plot(xrange(infer_iter_OPE4), map_values_OPE4, label='OPE4');
    plt.ylabel('MAP value');
#     plt.legend();

    plt.subplot(222)
    plt.plot(xrange(infer_iter_OPE), cosine_dis_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE1), cosine_dis_OPE1, label='OPE1');
    plt.plot(xrange(infer_iter_OPE2), cosine_dis_OPE2, label='OPE2');
    plt.plot(xrange(infer_iter_OPE3), cosine_dis_OPE3, label='OPE3');
    plt.plot(xrange(infer_iter_OPE4), cosine_dis_OPE4, label='OPE4');
    plt.ylabel('Cosine dis');
#     plt.legend();

    plt.subplot(223)
    plt.plot(xrange(infer_iter_OPE), euclid_dis_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE1), euclid_dis_OPE1, label='OPE1');
    plt.plot(xrange(infer_iter_OPE2), euclid_dis_OPE2, label='OPE2');
    plt.plot(xrange(infer_iter_OPE3), euclid_dis_OPE3, label='OPE3');
    plt.plot(xrange(infer_iter_OPE4), euclid_dis_OPE4, label='OPE4');
    plt.ylabel('Euclid dis');
    plt.xlabel('iteration');
#     plt.legend();
    
    plt.subplot(224)
    plt.plot(xrange(infer_iter_OPE), kl_divergence_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE1), kl_divergence_OPE1, label='OPE1');
    plt.plot(xrange(infer_iter_OPE2), kl_divergence_OPE2, label='OPE2');
    plt.plot(xrange(infer_iter_OPE3), kl_divergence_OPE3, label='OPE3');
    plt.plot(xrange(infer_iter_OPE4), kl_divergence_OPE4, label='OPE4');
    plt.ylabel('KL divergence');
    plt.xlabel('iteration');
    plt.legend();
    
    dis_file_name = output_folder +'/' + 'generated_doc_%d_dis.eps'%doc_id;
    plt.savefig(dis_file_name);
    plt.close();
    
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

def compute_kl_divergences(theta_true, thetas):
    iter_count = thetas.shape[0];
    values = np.zeros(iter_count);
    for i in xrange(iter_count):
        values[i] = compute_kl_divergence(theta_true, thetas[i]);
    return values;
    
def compute_kl_divergence(theta_true, theta):
    _sum = 0.
    for i in xrange(len(theta_true)):
        if(theta[i] != 0):
            _sum += theta_true[i] * np.log(theta_true[i] / theta[i]);
    return _sum;

if __name__ == '__main__':
    main()
