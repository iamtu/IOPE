
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
    os.makedirs(output_folder + '/same_theta')
    os.makedirs(output_folder + '/not_same_theta')
 
    print 'reading setting ...'
    settings = utilities.read_setting(setting_file)

    doc_count = 1000; # number of documents in comparison

    (test_words_ids, test_counts_ids) = utilities.read_data_for_MAP(test_data_folder);
    test_doc_ids = random.sample(range(0, len(test_words_ids)), doc_count);
 
#     init LDA model with beta file name
    # FIXME -     zeta = calculate_average_doc_length(beta_file_name);
    if 'nytimes' in beta_file_name:
        zeta = 330
    elif 'pubmed' in beta_file_name:
        zeta = 65
    else:
        print "not support this dataset"
        exit(1)
    
    lda_model = LDA(zeta, settings['alpha'], beta_file_name);
     
     
#    1. comparison of OPE vs OPE1234
#     simulating generating a document and do approximating a document
    print 'Simulation ...'
    print 'Generate %d docs to compare.'%doc_count
    for x in xrange(0, doc_count):
        (theta_true, word_ids, count_ids) = lda_model.generate_document()
        infer_generate_compare(lda_model, x, theta_true, True, word_ids, count_ids, output_folder);
        infer_generate_compare(lda_model, x, theta_true, False , word_ids, count_ids, output_folder);
         
#    infer test documents
    print 'Compare random %d docs in test data'%(len(test_doc_ids))
    for x in xrange(0, len(test_doc_ids)):
        infer_test_doc_compare(lda_model, test_doc_ids[x], True, test_words_ids[test_doc_ids[x]], test_counts_ids[test_doc_ids[x]], output_folder);
        infer_test_doc_compare(lda_model, test_doc_ids[x], False, test_words_ids[test_doc_ids[x]], test_counts_ids[test_doc_ids[x]], output_folder);
    

#    2. compare OPE vs OPE3 in terms of number of iterations

#     print 'Compare OPE vs OPE3 in terms of number of iterations'
#     print 'Generate %d docs to compare.'%doc_count
#     for x in xrange(0, doc_count):
#         (theta_true, word_ids, count_ids) = lda_model.generate_document()
#         infer_OPE_OPE3_number_interation_compare_generate(lda_model, x, theta_true, False , \
#                                                           word_ids, count_ids, output_folder);
# 
#     for x in xrange(0, len(test_doc_ids)):
#         infer_OPE_OPE3_number_interation_compare_test_doc(lda_model, test_doc_ids[x], False, \
#                                                           test_words_ids[test_doc_ids[x]], test_counts_ids[test_doc_ids[x]], output_folder);

    print 'Output result in folder %s' % output_folder

def infer_OPE_OPE3_number_interation_compare_test_doc(lda_model, doc_id, is_init_theta, word_ids, count_ids, output_folder):
    run_time = 5;
    infer_iter_OPE = 200;
    infer_iter_OPE3 = 100;
    
    init_theta = np.random.rand(lda_model._K) + 1.
    init_theta /= sum(init_theta)
    if is_init_theta == False:
        init_theta = None
    
    map_temps = np.zeros((run_time, infer_iter_OPE));
    for i in xrange(run_time):
        thetas_OPE = lda_model.OPE(word_ids, count_ids, init_theta, infer_iter_OPE);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    map_values_OPE = np.mean(map_temps, axis = 0);


    map_temps = np.zeros((run_time, infer_iter_OPE3));
    for i in xrange(run_time):
        thetas_OPE3 = lda_model.OPE3(word_ids, count_ids, init_theta, infer_iter_OPE3);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE3, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    map_values_OPE3 = np.mean(map_temps, axis = 0);

    
    
    plt.figure();
    plt.plot(xrange(infer_iter_OPE), map_values_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE3), map_values_OPE3, label='OPE3');
    plt.ylabel('MAP value');
    plt.legend();
    
    if is_init_theta == False:
        sub_folder = 'not_same_theta'
    else :
        sub_folder = 'same_theta'
    dis_file_name = output_folder +'/' + sub_folder +  '/test_doc_%d_dis.eps'%doc_id;
    plt.savefig(dis_file_name);
    plt.close();
    

def infer_OPE_OPE3_number_interation_compare_generate(lda_model, doc_id, theta_true, is_init_theta, word_ids, count_ids, output_folder):
    
    run_time = 5;
    infer_iter_OPE = 200;
    infer_iter_OPE3 = 100;
    
    init_theta = np.random.rand(lda_model._K) + 1.
    init_theta /= sum(init_theta)
    if is_init_theta == False:
        init_theta = None

    map_temps = np.zeros((run_time, infer_iter_OPE));
    cosine_temps = np.zeros((run_time, infer_iter_OPE));
    euclid_temps = np.zeros((run_time, infer_iter_OPE));
    kl_divergence_temps = np.zeros((run_time, infer_iter_OPE));
    
    for i in xrange(run_time):
        thetas_OPE = lda_model.OPE(word_ids, count_ids, init_theta, infer_iter_OPE);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
        cosine_temps[i] = compute_cosine_dis(theta_true, thetas_OPE);
        euclid_temps[i] = compute_euclid_dis(theta_true, thetas_OPE);
        kl_divergence_temps[i] = compute_kl_divergences(theta_true, thetas_OPE);
    map_values_OPE = np.mean(map_temps, axis = 0);
    cosine_dis_OPE = np.mean(cosine_temps, axis = 0);
    euclid_dis_OPE = np.mean(euclid_temps, axis = 0)
    kl_divergence_OPE = np.mean(euclid_temps, axis = 0)

    map_temps = np.zeros((run_time, infer_iter_OPE3));
    cosine_temps = np.zeros((run_time, infer_iter_OPE3));
    euclid_temps = np.zeros((run_time, infer_iter_OPE3));
    kl_divergence_temps = np.zeros((run_time, infer_iter_OPE3));

    for i in xrange(run_time):
        thetas_OPE3 = lda_model.OPE3(word_ids, count_ids, init_theta, infer_iter_OPE3);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE3, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
        cosine_temps[i] = compute_cosine_dis(theta_true, thetas_OPE3);
        euclid_temps[i] = compute_euclid_dis(theta_true, thetas_OPE3);
        kl_divergence_temps[i] = compute_kl_divergences(theta_true, thetas_OPE3);
    map_values_OPE3 = np.mean(map_temps, axis = 0);
    cosine_dis_OPE3 = np.mean(cosine_temps, axis = 0);
    euclid_dis_OPE3 = np.mean(euclid_temps, axis = 0);
    kl_divergence_OPE3 = np.mean(euclid_temps, axis = 0);

    plt.figure();

    plt.subplot(221)
    plt.plot(xrange(infer_iter_OPE), map_values_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE3), map_values_OPE3, label='OPE3');
    plt.ylabel('MAP value');

    plt.subplot(222)
    plt.plot(xrange(infer_iter_OPE), cosine_dis_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE3), cosine_dis_OPE3, label='OPE3');
    plt.ylabel('Cosine dis');

    plt.subplot(223)
    plt.plot(xrange(infer_iter_OPE), euclid_dis_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE3), euclid_dis_OPE3, label='OPE3');
    plt.ylabel('Euclid dis');
    plt.xlabel('iteration');
    
    plt.subplot(224)
    plt.plot(xrange(infer_iter_OPE), kl_divergence_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE3), kl_divergence_OPE3, label='OPE3');
    plt.ylabel('KL divergence');
    plt.xlabel('iteration');
    plt.legend();
    
                
    if is_init_theta == False:
        sub_folder = 'not_same_theta'
    else :
        sub_folder = 'same_theta'
    dis_file_name = output_folder +'/' + sub_folder + '/generated_doc_%d_dis.eps'%doc_id;
    plt.savefig(dis_file_name);


def infer_test_doc_compare(lda_model, doc_id, is_init_theta, word_ids, count_ids, output_folder):
    
    run_time = 5;
    infer_iter_OPE = 50;
    infer_iter_OPE1 = 50;
    infer_iter_OPE2 = 50;
    infer_iter_OPE3 = 50;
    infer_iter_OPE4 = 50;    
    
    init_theta = np.random.rand(lda_model._K) + 1.
    init_theta /= sum(init_theta)
    if is_init_theta == False:
        init_theta = None
    
    map_temps = np.zeros((run_time, infer_iter_OPE));
    for i in xrange(run_time):
        thetas_OPE = lda_model.OPE(word_ids, count_ids, init_theta, infer_iter_OPE);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    map_values_OPE = np.mean(map_temps, axis = 0);

    map_temps = np.zeros((run_time, infer_iter_OPE1));
    for i in xrange(run_time):
        thetas_OPE1 = lda_model.OPE1(word_ids, count_ids, init_theta, infer_iter_OPE1);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE1, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    map_values_OPE1 = np.mean(map_temps, axis = 0);

    map_temps = np.zeros((run_time, infer_iter_OPE2));
    for i in xrange(run_time):
        thetas_OPE2 = lda_model.OPE2(word_ids, count_ids, init_theta, infer_iter_OPE2);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE2, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    map_values_OPE2 = np.mean(map_temps, axis = 0);

    map_temps = np.zeros((run_time, infer_iter_OPE3));
    for i in xrange(run_time):
        thetas_OPE3 = lda_model.OPE3(word_ids, count_ids, init_theta, infer_iter_OPE3);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE3, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    map_values_OPE3 = np.mean(map_temps, axis = 0);

    map_temps = np.zeros((run_time, infer_iter_OPE4));
    for i in xrange(run_time):
        thetas_OPE4 = lda_model.OPE4(word_ids, count_ids, init_theta, infer_iter_OPE4, 0.01);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE4, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
    map_values_OPE4 = np.mean(map_temps, axis = 0);
    
    
    plt.figure();
    plt.plot(xrange(infer_iter_OPE), map_values_OPE, label='OPE');
    plt.plot(xrange(infer_iter_OPE1), map_values_OPE1, label='OPE1');
    plt.plot(xrange(infer_iter_OPE2), map_values_OPE2, label='OPE2');
    plt.plot(xrange(infer_iter_OPE3), map_values_OPE3, label='OPE3');
    plt.plot(xrange(infer_iter_OPE4), map_values_OPE4, label='OPE4');
    plt.ylabel('MAP value');
    plt.legend();
    
    if is_init_theta == False:
        sub_folder = 'not_same_theta'
    else :
        sub_folder = 'same_theta'
    dis_file_name = output_folder +'/' + sub_folder +  '/test_doc_%d_dis.eps'%doc_id;
    plt.savefig(dis_file_name);
    plt.close();

    txt_file_name = output_folder +'/' + sub_folder +  '/test_doc_%d_dis.txt'%doc_id;
    fpt = open(txt_file_name, 'w');
    fpt.write('map values\n');
    fpt.write(utilities.np_array_to_string(map_values_OPE));
    fpt.write(utilities.np_array_to_string(map_values_OPE1));
    fpt.write(utilities.np_array_to_string(map_values_OPE2));
    fpt.write(utilities.np_array_to_string(map_values_OPE3));
    fpt.write(utilities.np_array_to_string(map_values_OPE4));
    fpt.close();


def infer_generate_compare(lda_model, doc_id, theta_true, is_init_theta, word_ids, count_ids, output_folder):

    run_time = 5;
    infer_iter_OPE = 50;
    infer_iter_OPE1 = 50;
    infer_iter_OPE2 = 50;
    infer_iter_OPE3 = 50;
    infer_iter_OPE4 = 50;
    
    
    init_theta = np.random.rand(lda_model._K) + 1.
    init_theta /= sum(init_theta)
    if is_init_theta == False:
        init_theta = None
    
    #OPE
    map_temps = np.zeros((run_time, infer_iter_OPE));
    cosine_temps = np.zeros((run_time, infer_iter_OPE));
    euclid_temps = np.zeros((run_time, infer_iter_OPE));
    kl_divergence_temps = np.zeros((run_time, infer_iter_OPE));
    
    for i in xrange(run_time):
        thetas_OPE = lda_model.OPE(word_ids, count_ids, init_theta, infer_iter_OPE);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
        cosine_temps[i] = compute_cosine_dis(theta_true, thetas_OPE);
        euclid_temps[i] = compute_euclid_dis(theta_true, thetas_OPE);
        kl_divergence_temps[i] = compute_kl_divergences(theta_true, thetas_OPE);
    map_values_OPE = np.mean(map_temps, axis = 0);
    cosine_dis_OPE = np.mean(cosine_temps, axis = 0);
    euclid_dis_OPE = np.mean(euclid_temps, axis = 0)
    kl_divergence_OPE = np.mean(euclid_temps, axis = 0)
    
    #OPE1
    map_temps = np.zeros((run_time, infer_iter_OPE1));
    cosine_temps = np.zeros((run_time, infer_iter_OPE1));
    euclid_temps = np.zeros((run_time, infer_iter_OPE1));
    kl_divergence_temps = np.zeros((run_time, infer_iter_OPE1));

    for i in xrange(run_time):
        thetas_OPE1 = lda_model.OPE1(word_ids, count_ids, init_theta, infer_iter_OPE1);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE1, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
        cosine_temps[i] = compute_cosine_dis(theta_true, thetas_OPE1);
        euclid_temps[i] = compute_euclid_dis(theta_true, thetas_OPE1);
        kl_divergence_temps[i] = compute_kl_divergences(theta_true, thetas_OPE1);
    map_values_OPE1 = np.mean(map_temps, axis = 0);
    cosine_dis_OPE1 = np.mean(cosine_temps, axis = 0);
    euclid_dis_OPE1 = np.mean(euclid_temps, axis = 0);
    kl_divergence_OPE1 = np.mean(euclid_temps, axis = 0);

    #OPE2
    map_temps = np.zeros((run_time, infer_iter_OPE2));
    cosine_temps = np.zeros((run_time, infer_iter_OPE2));
    euclid_temps = np.zeros((run_time, infer_iter_OPE2));
    kl_divergence_temps = np.zeros((run_time, infer_iter_OPE2));
    for i in xrange(run_time):
        thetas_OPE2 = lda_model.OPE2(word_ids, count_ids, init_theta, infer_iter_OPE2);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE2, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
        cosine_temps[i] = compute_cosine_dis(theta_true, thetas_OPE2);
        euclid_temps[i] = compute_euclid_dis(theta_true, thetas_OPE2);
        kl_divergence_temps[i] = compute_kl_divergences(theta_true, thetas_OPE2);
    map_values_OPE2 = np.mean(map_temps, axis = 0);
    cosine_dis_OPE2 = np.mean(cosine_temps, axis = 0);
    euclid_dis_OPE2 = np.mean(euclid_temps, axis = 0);
    kl_divergence_OPE2 = np.mean(euclid_temps, axis = 0);

    #OPE3
    map_temps = np.zeros((run_time, infer_iter_OPE3));
    cosine_temps = np.zeros((run_time, infer_iter_OPE3));
    euclid_temps = np.zeros((run_time, infer_iter_OPE3));
    kl_divergence_temps = np.zeros((run_time, infer_iter_OPE3));

    for i in xrange(run_time):
        thetas_OPE3 = lda_model.OPE3(word_ids, count_ids, init_theta, infer_iter_OPE3);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE3, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
        cosine_temps[i] = compute_cosine_dis(theta_true, thetas_OPE3);
        euclid_temps[i] = compute_euclid_dis(theta_true, thetas_OPE3);
        kl_divergence_temps[i] = compute_kl_divergences(theta_true, thetas_OPE3);
    map_values_OPE3 = np.mean(map_temps, axis = 0);
    cosine_dis_OPE3 = np.mean(cosine_temps, axis = 0);
    euclid_dis_OPE3 = np.mean(euclid_temps, axis = 0);
    kl_divergence_OPE3 = np.mean(euclid_temps, axis = 0);

    #OPE4    
    map_temps = np.zeros((run_time, infer_iter_OPE4));
    cosine_temps = np.zeros((run_time, infer_iter_OPE4));
    euclid_temps = np.zeros((run_time, infer_iter_OPE4));
    kl_divergence_temps = np.zeros((run_time, infer_iter_OPE4));

    for i in xrange(run_time):
        thetas_OPE4 = lda_model.OPE4(word_ids, count_ids, init_theta, infer_iter_OPE1, 0.01);    
        map_temps[i] = lda_model.compute_MAPs(thetas_OPE4, lda_model._beta[:,word_ids], lda_model._alpha[0], count_ids);
        cosine_temps[i] = compute_cosine_dis(theta_true, thetas_OPE4);
        euclid_temps[i] = compute_euclid_dis(theta_true, thetas_OPE4);
        kl_divergence_temps[i] = compute_kl_divergences(theta_true, thetas_OPE4);
    map_values_OPE4 = np.mean(map_temps, axis = 0);
    cosine_dis_OPE4 = np.mean(cosine_temps, axis = 0);
    euclid_dis_OPE4 = np.mean(euclid_temps, axis = 0)
    kl_divergence_OPE4 = np.mean(euclid_temps, axis = 0)


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
    
                
    if is_init_theta == False:
        sub_folder = 'not_same_theta'
    else :
        sub_folder = 'same_theta'
    dis_file_name = output_folder +'/' + sub_folder + '/generated_doc_%d_dis.eps'%doc_id;
    plt.savefig(dis_file_name);
    plt.close();
    
    #save image as txt
    txt_file_name = output_folder +'/' + sub_folder + '/generated_doc_%d_dis.txt'%doc_id
    fpt = open(txt_file_name, 'w');
    fpt.write('map value \n');
    fpt.write(utilities.np_array_to_string(map_values_OPE));
    fpt.write(utilities.np_array_to_string(map_values_OPE1));
    fpt.write(utilities.np_array_to_string(map_values_OPE2));
    fpt.write(utilities.np_array_to_string(map_values_OPE3));
    fpt.write(utilities.np_array_to_string(map_values_OPE4));
    
    fpt.write('cosin distance\n');
    fpt.write(utilities.np_array_to_string(cosine_dis_OPE));
    fpt.write(utilities.np_array_to_string(cosine_dis_OPE1));
    fpt.write(utilities.np_array_to_string(cosine_dis_OPE2));
    fpt.write(utilities.np_array_to_string(cosine_dis_OPE3));
    fpt.write(utilities.np_array_to_string(cosine_dis_OPE4));
    
    fpt.write('euclid distance\n');
    fpt.write(utilities.np_array_to_string(euclid_dis_OPE));
    fpt.write(utilities.np_array_to_string(euclid_dis_OPE1));
    fpt.write(utilities.np_array_to_string(euclid_dis_OPE2));
    fpt.write(utilities.np_array_to_string(euclid_dis_OPE3));
    fpt.write(utilities.np_array_to_string(euclid_dis_OPE4));
    
    fpt.write('kl-divergence\n');
    fpt.write(utilities.np_array_to_string(kl_divergence_OPE));
    fpt.write(utilities.np_array_to_string(kl_divergence_OPE1));
    fpt.write(utilities.np_array_to_string(kl_divergence_OPE2));
    fpt.write(utilities.np_array_to_string(kl_divergence_OPE3));
    fpt.write(utilities.np_array_to_string(kl_divergence_OPE4));
    
    fpt.close();
    
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
