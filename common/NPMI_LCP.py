import sys, string
import numpy as np

def read_minibatch(fp, batch_size):
    wordids = list()
    for i in range(batch_size):
        line = fp.readline()
        # check end of file
        if len(line) < 5:
            break
        ids = list()
        terms = string.split(line)
        for i in range(1,int(terms[0]) + 1):
            term_count = terms[i].split(':')
            ids.append(int(term_count[0]))
        wordids.append(ids)
    return wordids

def parse_docs(corp):
    wordids = list()
    for line in corp:
        ids = list()
        terms = string.split(line)
        for i in range(1,int(terms[0]) + 1):
            term_count = terms[i].split(':')
            ids.append(int(term_count[0]))
        wordids.append(ids)
    return(wordids)

def read_topics(folder, i, j):
    filename = '%s/top10_%d_%d.dat' % (folder, i+1, j+1)
    topics = np.loadtxt(filename)
    topics = topics.astype(int)
    return(topics)

def read_loops(model_folder):
    loops_filename = '%s/loops.csv'%(model_folder)
    f = open(loops_filename, 'r')
    line = f.readline()
    ij = line.split(',')
    I = int(ij[0])
    J = int(ij[1])
    print'model folder: %s'%(model_folder)
    print'\t number of iters of training %d'%(I)
    print'\t number of minibatch %d'%(J)
    return(I, J)

def filter_topics(folder, i, j, filtered_tops):
    filename = '%s/top10_%d_%d.dat' % (folder, i+1, j+1)
    # get topics
    topics = np.loadtxt(filename)
    K = topics.shape[0]
    T = topics.shape[1]
    for k in range(K):
        for t in range(T):
            if not int(topics[k][t]) in filtered_tops:
                filtered_tops.append(int(topics[k][t]))

def co_dfreq_all(wordids, filtered_tops, num_terms, df, duv):
    tmp = list()
    for i in range(num_terms):
        if filtered_tops[i] in wordids:
            df[i] += 1
            tmp.append(i)
    n = len(tmp)
    if n < 2: return
    for i in range(n):
        for j in range(i+1, n):
            duv[tmp[i]][tmp[j]] += 1
            duv[tmp[j]][tmp[i]] += 1



def NPMI_LCP_calculator(train_file_name, output_folder, settings, tops):
    print 'calculating NPMI_LCP of %s, %s' %(train_file_name, output_folder)
    print 'reading training data...'
    wordids = list()
    datafp = open(train_file_name, 'r')
    while True:
        batch_word_ids = read_minibatch(datafp, settings['batch_size'])
        # Stop condition
        if len(batch_word_ids) == 0:
            break
        wordids = wordids + batch_word_ids
    datafp.close()

    # define constants
    N = len(wordids) # number of documents in training data
    logN = np.log(N)
    print 'number of training documents: %d' %N

    filtered_tops = list()
    print'reading loops ...'
    (I_ofw, J_ofw) = read_loops(output_folder)

    # filter topics
    print'filtering topics ...'
    for i in range(I_ofw):
        for j in range(J_ofw):
            filter_topics(output_folder, i, j, filtered_tops)
    filtered_tops.sort()
    num_terms = len(filtered_tops)
    print'\t number of terms in topics %d'%(num_terms)

    # compute co-frequencies
    print'computing co-frequencies ...'
    df = [0 for i in range(num_terms)]
    duv = [[] for i in range(num_terms)]
    for i in range(num_terms):
        duv[i] = [1 for j in range(num_terms)]
    for d in range(N):
        co_dfreq_all(wordids[d], filtered_tops, num_terms, df, duv)
        if d%10000 == 0: print d
    del wordids

    # compute coherence of topic models
    print'computing topic coherence of models ...'

    print output_folder
    # file names for writing LCP measures
    filename_mean_LCP = '%s/mean-LCP-coherence-top10.csv' % (output_folder)
    fmean_LCP = open(filename_mean_LCP, 'w')
    filename_median_LCP = '%s/median-LCP-coherence-top10.csv' % (output_folder)
    fmedian_LCP = open(filename_median_LCP, 'w')
    filename_coh_LCP = '%s/coherence-LCP-top10.csv' % (output_folder)
    # file names for writing NPMI measures
    filename_mean_NPMI = '%s/mean-NPMI-coherence-top10.csv' % (output_folder)
    fmean_NPMI = open(filename_mean_NPMI, 'w')
    filename_median_NPMI = '%s/median-NPMI-coherence-top10.csv' % (output_folder)
    fmedian_NPMI = open(filename_median_NPMI, 'w')
    filename_coh_NPMI = '%s/coherence-NPMI-top10.csv' % (output_folder)
    for i in range(I_ofw):
        for j in range(J_ofw):
            print 'loop %d minibatch %d' % (i+1, j+1)
            # Read the learned output_folder
            print '\t reading output_folder...'
            list_top = read_topics(output_folder, i, j)
            K = list_top.shape[0]
            T = list_top.shape[1]
            #Compute coherence for topics
            print '\t computing coherence...'
            ch_LCP  = []
            ch_NPMI = []
            for k in range(K):
                # get indices of terms in filtered topic list
                index = []
                for t in range(T):
                    index.append(filtered_tops.index(list_top[k][t]))
                # compute
                total_LCP = 0.; total_NPMI = 0.
                for ii in range(1,T):
                    for jj in range(0, ii):
                        if(df[index[jj]] > 0):
                            total_LCP += np.log(float(duv[index[ii]][index[jj]]) / df[index[jj]])
                        if(duv[index[ii]][index[jj]]) != 0:
                            total_NPMI += -1. + (np.log(df[index[ii]] * df[index[jj]]) - 2*logN) / (np.log(duv[index[ii]][index[jj]]) - logN)
                ch_LCP.append(total_LCP); ch_NPMI.append(total_NPMI)
            print 'LCP %f' % np.mean(ch_LCP)
            print 'NPMI %f' % np.mean(ch_NPMI)
            fmean_LCP.writelines('%f,' % np.mean(ch_LCP))
            fmean_NPMI.writelines('%f,' % np.mean(ch_NPMI))
            fmedian_LCP.writelines('%f,' % np.median(ch_LCP))
            fmedian_NPMI.writelines('%f,' % np.median(ch_NPMI))
            coh_LCP = open(filename_coh_LCP, 'a')
            coh_LCP.writelines('%f,' % item for item in ch_LCP)
            coh_LCP.writelines('\n')
            coh_LCP.close()
            coh_NPMI = open(filename_coh_NPMI, 'a')
            coh_NPMI.writelines('%f,' % item for item in ch_NPMI)
            coh_NPMI.writelines('\n')
            coh_NPMI.close()
    fmean_LCP.close()
    fmean_NPMI.close()
    fmedian_LCP.close()
    fmedian_NPMI.close()

    print'done!!!'
