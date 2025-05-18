import math
from stemming.porter2 import stem

# Task 1
# Design a Pseudo Relevance Model to rank documents by using an initial query Q and then generate a training set.\n",

# 1. Design a python function `bm25(col, q, df)` to calculate a BM25 score for all documents in the \"Training_set\", where `coll` is the output of `coll.parse_rcv_coll(coll_fname, stop_words)`, see week 9 solution if you do not know this function; q is a query (e.g. \"Convicts, repeat offenders\"), and df is a dictionary of term document-frequency pairs

def avg_doc_len(coll):
    sum_doc_length = 0
    for id, doc in coll.get_docs().items():
        doc_size = doc.get_doc_len()        
        sum_doc_length += doc_size

    avg_doc_length = sum_doc_length / coll.get_num_docs()

    return avg_doc_length

def bm25(col, q, df):
    bm25s = {}
    avg_dl = avg_doc_len(col)
    no_docs = col.get_num_docs()

    for id, doc in col.get_docs().items():
        query_terms = q.split()
        qfs = {}

        for t in query_terms:
            term = stem(t.lower())

            try:
                qfs[term] += 1
            except KeyError:
                qfs[term] = 1 

        k = 1.2 * ((1 - 0.75) + 0.75 * (doc.get_doc_len() / float(avg_dl)))
        bm25_ = 0.0
            
        for qt in qfs.keys():
            n = 0

            if qt in df.keys():
                n = df[qt]
                f = doc.get_term_count(qt)
                qf = qfs[qt]
                bm = math.log(1.0 / ((n + 0.5) / (no_docs - n + 0.5)), 2) * (((1.2 + 1) * f) / (k + f)) * ( ((100 + 1) * qf) / float(100 + qf))
                
                bm25_ += bm
        print('bm25_ = ', bm25_)
        
        bm25s[doc.get_docid()] = bm25_

    return bm25s

# 2. Call the `bm25()` function in the main function and save the result to the text file `PRModel_R102.dat`. Each line contains the document number and the corresponding BM25 score and they are sorted in descending order

# 3. Extend the main function to generate a training set D which includes both D+ (positive and likely relevant documents) and D- (negative and likely irrelevant documents) in the given unlabelled document set (U). The output of the function is a file \"PTraining_benchmark.txt\"\n"

# 4. Re-run the week 10 solution (the two .py files) by replacing \"Training_benchmark.txt\"
# with "PTraining_benchmark.txt", you may get the following output:


if __name__ == "__main__":
    import sys
    import os
    import coll
    import df
    import math

    print('Current Directory: ', os.getcwd())
    os.chdir('./week10')
    print('Current Directory: ', os.getcwd())
    
    coll_fname = "Training_set"
    stopwords_file = open('common-english-words.txt', 'r')

    stop_words = stopwords_file.read().split(',')
    stopwords_file.close()

    coll_ = coll.parse_rcv_coll(coll_fname, stop_words)
    df_ = df.calc_df(coll_)

    
    # Calling the BM25 function for Task 1.2
    bm25_1 = bm25(coll_, 'Convicts repeat offenders', df_)
    
    print('For query Q = ' + "\"Convicts repeat offenders\"")
    os.chdir('..')
    wFile = open('PRModel_R102.dat', 'a')

    for (k, v) in sorted(bm25_1.items(), key=lambda x: x[1], reverse=True):
        wFile.write(k +' '+ str(v) +'\n')
    wFile.close()

    print('BM25 scores saved to PRModel_R102.dat')

# Task 1.3
writeFile = open('PTraining_benchmark.txt', 'a')
bm25_threshold = 1.0
dataFile = open('PRModel_R102.dat')

file_ = dataFile.readlines()

for line in file_:
    line = line.strip()
    line_string = line.split()

    if float(line_string[1]) > bm25_threshold:
        writeFile.write('R102 ' + line_string[0] + ' 1' +'\n')
    else:
        writeFile.write('R102 ' + line_string[0] + ' 0' +'\n')
writeFile.close()
dataFile.close()