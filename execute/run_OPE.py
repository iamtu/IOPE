import sys
sys.path.insert(0, '../')
from algorithm.mlope import MLOPE, MLOPE1, MLOPE2, MLOPE3, MLOPE4
from algorithm.onlineope import OnlineOPE, OnlineOPE1, OnlineOPE2, OnlineOPE3, OnlineOPE4

from common import utilities

class runOPE:
    def __init__(self, algo_name, train_file_name, settings, output_folder, test_data, top_words_count):
        self.train_file_name = train_file_name
        self.settings = settings
        self.output_folder = output_folder
        self.test_data = test_data
        self.top_words_count = top_words_count
        self.algo_name = algo_name

    def run(self):
        if self.algo_name == 'ml-ope':
            model = MLOPE(self.settings['num_terms'], self.settings['num_topics'],
                    self.settings['alpha'], self.settings['tau0'],
                    self.settings['kappa'], self.settings['iter_infer'],
                    self.settings['p_bernoulli'])
        elif self.algo_name == 'ml-ope1':
            model = MLOPE1(self.settings['num_terms'], self.settings['num_topics'],
                        self.settings['alpha'], self.settings['tau0'],
                        self.settings['kappa'], self.settings['iter_infer'],
                        self.settings['p_bernoulli'])
        elif self.algo_name == 'ml-ope2':
            model = MLOPE2(self.settings['num_terms'], self.settings['num_topics'],
                        self.settings['alpha'], self.settings['tau0'],
                        self.settings['kappa'], self.settings['iter_infer'],
                        self.settings['p_bernoulli'])
        elif self.algo_name == 'ml-ope3':
            model = MLOPE3(self.settings['num_terms'], self.settings['num_topics'],
                        self.settings['alpha'], self.settings['tau0'],
                        self.settings['kappa'], self.settings['iter_infer'],
                        self.settings['p_bernoulli'])
        elif self.algo_name == 'ml-ope4':
            model = MLOPE4(self.settings['num_terms'], self.settings['num_topics'],
                        self.settings['alpha'], self.settings['tau0'],
                        self.settings['kappa'], self.settings['iter_infer'],
			            self.settings['p_bernoulli'], self.settings['weighted_new4'])
        elif self.algo_name == 'online-ope':
            model = OnlineOPE(self.settings['num_docs'], self.settings['num_terms'],
                              self.settings['num_topics'], self.settings['alpha'],
                              self.settings['eta'], self.settings['tau0'],
                              self.settings['kappa'], self.settings['iter_infer'],
                              self.settings['p_bernoulli'])
        elif self.algo_name == 'online-ope1':
            model = OnlineOPE1(self.settings['num_docs'], self.settings['num_terms'],
                          self.settings['num_topics'], self.settings['alpha'],
                          self.settings['eta'], self.settings['tau0'],
                          self.settings['kappa'], self.settings['iter_infer'],
                          self.settings['p_bernoulli'])
        elif self.algo_name == 'online-ope2':
            model = OnlineOPE2(self.settings['num_docs'], self.settings['num_terms'],
                              self.settings['num_topics'], self.settings['alpha'],
                              self.settings['eta'], self.settings['tau0'],
                              self.settings['kappa'], self.settings['iter_infer'],
                              self.settings['p_bernoulli'])
        elif self.algo_name == 'online-ope3':
            model = OnlineOPE3(self.settings['num_docs'], self.settings['num_terms'],
                              self.settings['num_topics'], self.settings['alpha'],
                              self.settings['eta'], self.settings['tau0'],
                              self.settings['kappa'], self.settings['iter_infer'],
                              self.settings['p_bernoulli'])
        elif self.algo_name == 'online-ope4':
            model = OnlineOPE4(self.settings['num_docs'], self.settings['num_terms'],
                              self.settings['num_topics'], self.settings['alpha'],
                              self.settings['eta'], self.settings['tau0'],
                              self.settings['kappa'], self.settings['iter_infer'],
                              self.settings['p_bernoulli'], self.settings['weighted_new4'])
        else:
            pass
        # Start
        print'start!!!'
        is_belong_to_ml = False
        if(self.algo_name in ['ml-ope', 'ml-ope1','ml-ope2','ml-ope3','ml-ope4']):
            is_belong_to_ml = True

        i = 0
        while i < self.settings['iter_train']:
            i += 1
            print'\n***iter_train:%d***\n'%(i)
            datafp = open(self.train_file_name, 'r')
            j = 0
            while True:
                j += 1
                (wordids, wordcts) = utilities.read_minibatch_list_frequencies(datafp, self.settings['batch_size'])
                # Stop condition
                if len(wordids) == 0:
                    break

                print'---num_minibatch:%d---'%(j)
                (time_e, time_m, theta) = model.static_online(wordids, wordcts) # theta = array(batchsize*K)

                sparsity = utilities.compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')

                if is_belong_to_ml:
                    # Compute perplexities
                    LD2 = utilities.compute_perplexities_vb(model.beta, self.settings['alpha'], self.settings['eta'],
                                                            self.settings['iter_infer'], self.test_data)
                    # Search top words of each topics
                    top_words = utilities.list_top(model.beta, self.top_words_count)

                    # i : iter_train mormaly 1 , j : minibatch number
                    utilities.write_file(i, j, model.beta, time_e, time_m, theta, sparsity, LD2,
                                         top_words, self.top_words_count,self.output_folder)
                else:
                    LD2 = utilities.compute_perplexities_vb(model._lambda, self.settings['alpha'], self.settings['eta'],
                                                            self.settings['iter_infer'], self.test_data)
                    top_words = utilities.list_top(model._lambda, self.top_words_count)
                    utilities.write_file(i, j, model._lambda, time_e, time_m, theta, sparsity, LD2, top_words, self.top_words_count,
                                         self.output_folder)

            datafp.close()

        # Write settings
        print'write setting ...'
        file_name = '%s/setting.txt'%(self.output_folder)
        utilities.write_setting(self.settings, file_name)
        # Write final model to file
        print'write final model ...'
        file_name = '%s/beta_final.dat'%(self.output_folder)
        if is_belong_to_ml:
            utilities.write_topics(model.beta, file_name)
        else:
            utilities.write_topics(model._lambda, file_name)

        # Finish
        print'done!!!'
