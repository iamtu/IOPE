import sys
def stats(filename):
    token_count = 0
    doc_count = 0
    with open(filename) as f:
        for line in f:
            doc_count +=1 
            tokens = line.split()
            doc_length = int(tokens[0])
            for j in range(1, doc_length):
                token_count_pair = tokens[j].split(':')
                token_count_pair = map(int, token_count_pair)
                token_count += token_count_pair[1]
    f.close()
    print 'Tokens count : %d' % token_count
    print 'doc_length_averaage %f' % (1.0 * token_count / doc_count)
    return

if __name__ == '__main__':
    stats(sys.argv[1]);
