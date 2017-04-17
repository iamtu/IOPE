from numpy import genfromtxt
def aggregate(method, dataset, p_setting, measurement):

    if measurement == 'NPMI':
        measurement_file = 'mean-NPMI-coherence-top10.csv'
    elif measurement == 'perplexity':
        measurement_file = 'perplexities_1.csv'
    else:
        print "Error: Unknown measurement"

    timerun1_filename = '../models/%s/%s/%s/timerun1/%s'%(method, dataset, p_setting,measurement_file)
    timerun2_filename = '../models/%s/%s/%s/timerun2/%s'%(method, dataset, p_setting,measurement_file)
    timerun3_filename = '../models/%s/%s/%s/timerun3/%s'%(method, dataset, p_setting,measurement_file)
    timerun4_filename = '../models/%s/%s/%s/timerun4/%s'%(method, dataset, p_setting,measurement_file)
    timerun5_filename = '../models/%s/%s/%s/timerun5/%s'%(method, dataset, p_setting,measurement_file)

    timerun1 = genfromtxt(timerun1_filename, delimiter=',')
    timerun1 = timerun1[:-1].copy()
    timerun2 = genfromtxt(timerun2_filename, delimiter=',')
    timerun2 = timerun2[:-1].copy()
    timerun3 = genfromtxt(timerun3_filename, delimiter=',')
    timerun3 = timerun3[:-1].copy()
    timerun4 = genfromtxt(timerun4_filename, delimiter=',')
    timerun4 = timerun4[:-1].copy()
    timerun5 = genfromtxt(timerun5_filename, delimiter=',')
    timerun5 = timerun5[:-1].copy()
    average_run = (timerun1 + timerun2 + timerun3 + timerun4 + timerun5) / 5
    print average_run

    output_filename = '../models/%s/%s/%s/mean_%s.cvs'%(method, dataset, p_setting, measurement)
    print output_filename
    f = open(output_filename, 'w')
    for i in average_run:
        f.write('%.10f,'%(i))
    f.close()
    return

aggregate('ml-ope', 'ap', 'p_0.3', 'perplexity');
aggregate('ml-ope', 'ap', 'p_0.3', 'NPMI');
