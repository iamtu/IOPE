def count_token(filename):
    count = 0;
    with open(filename) as f:
        for line in f:
            tokens = line.split()
            doc_length = int(tokens[0])
            for j in range(1, doc_length):
                token_count_pair = tokens[j].split(':')
                token_count_pair = map(int, token_count_pair)
                count += token_count_pair[1]
    f.close()
    return (count)


if __name__ == '__main__':
    print "Counting number of tokens in datasets"
    print "ap: %d"%(count_token('../data/ap/ap_train.txt'))
    print "nyt: %d"%(count_token('../data/nyt/nyt_train.txt'))
    print "pubmed: %d"%(count_token('../data/pubmed/pubmed_train.txt'))
