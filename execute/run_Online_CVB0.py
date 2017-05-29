import sys
sys.path.insert(0, '../')
from algorithm.onlinecvb0 import OnlineCVB0
from common import utilities

class runOnlineCVB0:

    def __init__(self, train_file, settings, model_folder, test_data, tops):
        self.train_file = train_file
        self.settings = settings
        self.model_folder = model_folder
        self.test_data = test_data
        self.tops = tops

    def run(self):
        # Initialize the algorithm
        print'initialize the algorithm ...'
        online_cvb0 = OnlineCVB0(self.settings['num_tokens'], self.settings['num_terms'],
            self.settings['num_topics'], self.settings['alpha'], self.settings['eta'],
            self.settings['tau_phi'], self.settings['kappa_phi'], self.settings['s_phi'],
            self.settings['tau_theta'], self.settings['kappa_theta'],
            self.settings['s_theta'],self.settings['burn_in'])
        # Start
        print'start!!!'
        i = 0; j = 0
        while i < self.settings['iter_train']:
            i += 1
            print'\n***iter_train:%d***\n'%(i)
            datafp = open(self.train_file, 'r')
            while True:
                j += 1
                (wordtks, lengths) = utilities.read_minibatch_list_sequences(datafp, self.settings['batch_size'])
                # Stop condition
                if len(lengths) == 0:
                    j -= 1
                    break
                #
                print'---num_minibatch:%d---'%(j)
                (time_e, time_m, theta) = online_cvb0.static_online(wordtks, lengths)
                # Compute sparsity
                ##sparsity = utilities.compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')
                sparsity = 1.0
                # Compute perplexities
                pers = utilities.compute_perplexities_vb(online_cvb0.N_phi, self.settings['alpha'], self.settings['eta'],
                                                        self.settings['iter_infer'], self.test_data)
                # Search top words of each topics
                list_tops = utilities.list_top(online_cvb0.N_phi, self.tops)
                # Write files
                utilities.write_file(i, j, online_cvb0.N_phi, time_e, time_m, theta, sparsity, pers, list_tops, self.tops,
                                     self.model_folder)
            datafp.close()
        # Write settings
        print'write setting ...'
        file_name = '%s/setting.txt'%(self.model_folder)
        utilities.write_setting(self.settings, file_name)
        # Write final model to file
        print'write final model ...'
        file_name = '%s/beta_final.dat'%(self.model_folder)
        utilities.write_topics(online_cvb0.N_phi, file_name)
        # Finish
        print'done!!!'
