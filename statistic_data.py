import sys

def statistics(filename):
    print('count dimension and instance number of %s' % (filename))
    doc_count = 0;
    token_count = 0;
    min_index = 1000000;
    max_index = -1;
    with open(filename) as f:
        for line in f:
            doc_count += 1;
            tokens = line.split()
            for j in range(1, len(tokens)):
                word_pairs = tokens[j].split(':')
                index = int(word_pairs[0])
                token_count += int(word_pairs[1]);

                if min_index > index:
                    min_index = index
                if max_index < index:
                    max_index = index

    f.close()

    print 'number of documents %d' % (doc_count)
    print 'number of tokens %d' % (token_count)
    print ('max_index : %d' % (max_index))
    print ('min_index : %d' % (min_index))
    return


if __name__ == '__main__':
    statistics(sys.argv[1]);
