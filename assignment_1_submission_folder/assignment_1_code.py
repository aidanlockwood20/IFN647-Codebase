
# # IFN647 Assignment 1
# - Student Number: n10515402
# - First Name: Aidan
# - Last Name: Lockwood
# 


# Loading in the required packages


import string

import numpy as np
from stemming.porter2 import stem
import os 


# # Question 1. Parsing of Documents & Queries 
# The motivation for Question 1 is to design your own document and query parsers. So please don't use python packages that we didn't use in the workshop. 
# 
# ## Task 1.1:
# Define a document parsing function `parse_docs(stop_words, inputfolder)` to parse a data collection (e.g. RCV1v3 dataset), where parameter `stop_words` is a list of common English words (you may use the file `common-english-words.txt` to find all stop words), and parameter `inputfolder` is the folder that stores the set of XML files.
# 
# The following are the major steps in the document parsing function:
# ### Step 1 
# The function reads XML files from inputfolder (e.g., RCV1v3). For each XML file, it finds the document ID and index terms and then represents it in a DocV3 Object. 
# You need to define a DocV3 class by using Bag-of-Words to represent a document:
# - DocV3 needs a document ID variable (attribute newsID), which is simply assigned by the value of ‘itemid’ in <newsitem …> of the XML file.
# - In this task, `DocV3` can be initialized with three attributes: `newsID` attribute, an empty dictionary (the attribute name is terms) of key-value pair of (String term: int frequency); and doc_size (the document length) attribute.
# - You may define your own methods, e.g., `getNewsID()` to get the document ID, get_termList() to get a sorted list of all terms occurring in the document, etc.
# 
# ### Step 2 
# It then builds up a collection of `DocV3` objects for the given dataset, this collection can be a dictionary structure (as we used in the workshop), a linked list, or a class Rcv1Coll for storing a collection of `DocV3` objects. Please note the rest descriptions are based on the dictionary structure with newsID as key and DocV3 object as value.
# 
# ### Step 3 
# At last, it returns the collection of DocV3 objects.
# You also need to follow the following specification to define this parsing function:
# - Please use the basic text pre-processing steps, such as tokenizing, stopping words
# removal and stemming of words.
# Tokenizing –
# - You are required to provide definitions of words and terms and describe them as
# comments in your Python solution.
# - You need to tokenize at least the ‘<text>…</text>’ part of document, exclude all
# tags, and discard punctuations and/or numbers based on your definition of terms.
# - Define method add_term() for class DocV3 to add new term or increase term
# frequency when the term occur again.
# Stopping words removal and stemming of terms –
# - Use the given stopping words list (“common-english-words.txt”) to ignore/remove
# all stopping words. Open and read the given file of stop-words and store them into a
# list stopwordList. When adding a term, please check whether the term exists in the
# stopwordList, and ignore it if it is in the stopwordList. Note that you can update the
# "common-english-words.txt" file based on your data exploration of the dataset, but
# you will need to comment the change in your solution.
# - You can use porter2 stemming algorithm or other algorithms to update DocV3’s
# terms.


# Utility Functions Used within the assignment
file_dir = os.getcwd()

def get_docid(file_):

    for line in file_:
        line = line.strip()

        if line.startswith("<text>"):
            start_end = True
        
        if line.startswith("<newsitem "):
            for part in line.split():
                if part.startswith('itemid='):
                    docid = part.split('=')[1].split('/')[0]
                    docid = docid.replace('"', '')

    return docid

def get_stop_words(stop_words):
    
    stop_words_file = open(f'{file_dir}/assignment_1_submission_folder/{stop_words}', 'r')
    stop_words_list = stop_words_file.readlines()
    
    stop_words_file.close()

    stop_words_list = stop_words_list[0].split(',')
    return stop_words_list

def get_file_contents(file_name):  
    
    opened_file = open(file_name, 'r')
    contents = opened_file.readlines()
    opened_file.close()

    return contents

def parse_text(file_contents):

    start_end = False
    parsed_text = []

    for line in file_contents:
        line = line.strip()

        if line.startswith("<text>"):
            start_end = True

        if line.startswith('<p>'):
            line = line.replace('<p>', '').replace('</p>', '')
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = line.replace('quot', '')

        elif line.startswith('</text>'):
            start_end = False
        
        if start_end:
            parsed_text.append(line)
        

    return parsed_text

def write_task_file(contents, question_number):
    write_to_file = open(f'assignment_1_submission_folder/Aidan Lockwood_{question_number}.txt', 'w')
    for line in contents:
        write_to_file.write(line)
    
    write_to_file.close()



# DocV3 Class (with a linked list class to store them)


class DocV3:
    def __init__(self, itemid, next = None):
        self.newsid = itemid
        self.terms = {}
        self.doc_size = 0

        self.next = next
    
    def set_news_id(self, itemid):
        self.newsid = itemid
        return self.newsid
    
    def get_news_id(self):
        return self.newsid

    def set_doc_size(self, word_count): # mutator method for Task 3.1
        self.doc_size = word_count
        return self.doc_size
    
    def get_doc_size(self): # accessor method for Task 3.1
        return self.doc_size

    def get_terms_list(self):        
        return self.terms
    

    def add_terms_list(self, terms_list):
        self.terms = terms_list  
        return self.terms
    
    
class List_Docs:
    def __init__(self, hnode = None):
        self.head = hnode

    def insert(self, nnode):
        if not self.head:
            self.head = nnode
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = nnode

        return nnode

    # Need to add a printing function here    
    def print_list(self):

        printed_list = []
        current = self.head
        while current:
            print(f'Document {current.newsid} contains {len(current.terms)} terms, with a total word count of {current.doc_size} words')
            printed_list.append(f'Document {current.newsid} contains {len(current.terms)} indexing terms, with a total word count of {current.doc_size} words')

            for term in current.terms:
                print(f'{term} : {current.terms[term]}')
                printed_list.append(f'{term} : {current.terms[term]}')
            current = current.next
            
        return printed_list
    
    def get_list(self):
        current = self.head
        doc_list = []
        while current:
            doc_list.append(current)
            current = current.next

        return doc_list
    
    def get_list_size(self):
        current = self.head

        size = 0

        while current:
            current = current.next
            size += 1

        return size


# Main Parsing Function


def parse_docs(stop_words, inputfolder):
    stop_words_list = get_stop_words(stop_words)

    input_folder_files = os.listdir(f'assignment_1_submission_folder/{inputfolder}')

    # list of DocV3 objects - this will turn into a linked list
    docs_list = List_Docs()

    for path in input_folder_files:
        opened_file = get_file_contents(f'assignment_1_submission_folder/{inputfolder}{path}')

        current_doc = {}

        word_count = 0

        for line in opened_file:
            
            # Retrieve the docid and initialise a new instance of DocV3
            docid = get_docid(opened_file)
            doc_obj = DocV3(docid)

        parsed_text = parse_text(opened_file)

        split_text = []

        for line in parsed_text:
            for word in line.split():
                word_count += 1

                if word.lower() not in stop_words_list and not word.isdigit():
                    word = word.lower()
                    split_text.append(stem(word))
        
        split_text.pop(0)      

        for word in split_text:
            if word not in current_doc:
                current_doc[word] = 1
            else:
                current_doc[word] += 1

        # Order the current_doc dictionary by descending frequency
        current_doc = dict(sorted(current_doc.items(), key=lambda item: item[1], reverse=True))
        
        doc_obj.add_terms_list(current_doc)
        doc_obj.set_doc_size(word_count)

        docs_list.insert(doc_obj)        

    return docs_list


document_folder = 'RCV1v3/'

docs_v3_list = parse_docs('common-english-words.txt', document_folder)

docs_v3_list.print_list()


# ## Task 1.2 
# Define a query parsing function `Parse_Q(query, stop_words)`, where we assume the original query is a simple sentence or a title in a String format (<i>query</i>), and stop_words is a list of stop words that you can get from 'common-english-words.txt.
# 
# For example, let query = 
# 
# 'US EPA ranks Geo Metro car most fuel-efficient 1998 car.'
# 
# This function will return a dictionary:
# { 'epa' : 1, 'rank' : 1, 'geo' : 1, 'metro' : 1, 'car' : 2, 'fuel' : 1, 'effici' : 1}
# 
# Please note that you should use the same text transformation technique as the document, i.e., tokenising steps for queries **must be identical** to steps for documents and use three different queries to test the function.


def parse_q(query, stop_words):

    query_dict = {}

    stop_words_list = get_stop_words(stop_words)
    
    query = query.strip()

    punctuation = string.punctuation.replace('-', '')

    query = query.translate(str.maketrans('', '', punctuation))

    for word in query.split():

        hiphen_split = word.split('-')

        for compound_word in hiphen_split:
            if compound_word.lower() not in stop_words_list and not compound_word.isdigit():
                compound_word = stem(compound_word.lower())

                if compound_word not in query_dict:
                    query_dict[compound_word] = 1
                else:
                    query_dict[compound_word] += 1
        
    return query_dict


# ## Task 1.3
# Define a main function to test function `Parse_Docs()` and `Parse_Q()`. The main function uses the provided dataset, calls function `Parse_Docs()` to get a collection of `DocV3` objects. For each document in the collection, firstly print out its `newsID`, the number of terms and the total number of words in the document (<i>doc_size</i>). It then sorts the terms (by frequency) and prints out a <i>term:freq</i> list. At last, it saves the output into a text file (file name is: "your full name_Q1.txt).


def test_parsing(queries, stop_words):
    docs_v3_list = parse_docs(stop_words, 'RCV1v3/')
    file_contents = []


    printed_docs_v3_list = docs_v3_list.print_list()

    for line in printed_docs_v3_list:
        file_contents.append(line + '\n')
        
    for index, query in enumerate(queries):
        parsed_query = parse_q(query, stop_words)
        file_contents.append(f'Query: {query}')
        print(f'Parsed Query {index + 1}: \n{parsed_query}\n')
        file_contents.append(f'The parsed query {index + 1}: \n{parsed_query}\n\n')

    write_task_file(file_contents, 'Q1')
     


query_1 = 'US EPA ranks Geo Metro car most fuel-efficient 1997 car.'
query_2 = 'The quick brown fox jumped over the lazy dog.'
query_3 = 'In April 2025, NASA announced new details about its upcoming Artemis III mission to the Moon.'

test_queries = [query_1, query_2, query_3]

test_parsing(test_queries, 'common-english-words.txt')





# # Question 2: TF*IDF-based IR Model
# TF*IDF is a popular term weighting method, which uses the following Eq. (1) to calculate a weight for term k in a document i, where the base of log is e. You may review lecture notes to get the meaning of each variable in the equation.

# 
# ### Task 2.1
# Define a function `df(coll)`to calculate document-frequency for a given `DocV3` collection `coll` and return a {term:df, ...} dictionary. You need to test this function and produce a similar output as shown below.


def my_df(coll):
    term_freq = {}

    for doc in coll.get_list():
        for term in doc.terms:
            try:
                term_freq[term] += 1
            except KeyError:
                term_freq[term] = 1

    print(f'There are {coll.get_list_size()} documents in this data set and contains {len(term_freq)} unique terms.')
    
    term_freq = dict(sorted(term_freq.items(), key=lambda item: item[1], reverse=True))
    
    return term_freq


my_df(docs_v3_list)


# ## Task 2.2
# Use Eq. (1) to define a function `tfidf(doc, d_f, ndocs)` to calculate TF*IDF value (weight) of every term in a DocV3 object or a dictionary of {term:freq, ...} d_f is a {term:df, ...} dictionary and ndocs is the number of documents in a given `DocV3` collection. The function returns a {term: tfidf_weight, ...} dictionary for the given document doc.


def my_tfidf(doc, d_f, ndocs):
    # The normalised tfidf scores
    d_ik_scores = {}

    # To hold the terms with the docids in which they appear
    coll_frequency_dict = {}

    # For the collection: Indexing the lists, finding each term and counting the frequency of documents each term appears in
    for document in d_f.get_list():
        docid = document.newsid

        for term in document.terms:
            if term not in coll_frequency_dict:
                coll_frequency_dict[term] = {docid : document.terms[term]}
            else:
                coll_frequency_dict[term][docid] = document.terms[term]
    
    n_k_values = {}

    # This is the n_k values for each term in the collection
    for term in coll_frequency_dict:
        n_k_values[term] = len(coll_frequency_dict[term])
    
    f_ik_values = {}

    # For the document: Indexing the document and getting the frequency
    for term in doc.get_terms_list():
        if term not in f_ik_values:
            f_ik_values[term] = doc.terms[term]
        else:
            f_ik_values[term] += doc.terms[term]

    # Computing d_ik for each term in the dictionary
    for term in doc.get_terms_list():
        if term in coll_frequency_dict:
            d_ik = (np.log(f_ik_values[term]) + 1) * np.log(ndocs / n_k_values[term])
            # Normalise the d_ik score
            d_ik = d_ik / np.sqrt(sum([((np.log(f_ik_values[term_t]) + 1) * np.log(ndocs / n_k_values[term_t])) ** 2 for term_t in doc.get_terms_list()]))

            d_ik_scores[term] = d_ik
        else:
            d_ik_scores[term] = 0


    return d_ik_scores


d_f = parse_docs('common-english-words.txt', document_folder)
test_doc = d_f.get_list()[0]

# for term in test_doc:
#     print(term)
my_tfidf(test_doc, d_f, d_f.get_list_size())


# ## Task 2.3
# Define a main function to call <i>tfidf()</i> and **print out top 20 terms** (with its value of tf*idf weight) for each document in <i>RCV1v3</i> if it has more than 20 terms and save the output into a text file (file name is "your full name_Q2.txt")
# 
# - You also need to implement a TF*IDF based IR model
# - You can assume titles of XML document (the <title>...</title> part) are the original queries, and test at least three titles
# - You need to use function `Parse_Q()` that you defined for Question 1 to parse original queries 
# - For each query <i>Q</i>, please use the abstract model of ranking to calculate a ranking score for each document <i>D</i>.
# 
# This task does not require you to use cosine similarity to rank documents for a given query <i>Q</i>, but you do need to provide your observations to see whether the ranking results using cosine are similar to the ranking results using the abstract ranking model. **Hint:** you may calculate the magnitude for each document vector <i>D</i>.
# 
# At last, append the output (in descending order) into the text file ("your full name_Q2.txt")
# 


def main_function_q2(d_f, Q):
    # Call the tfidf function to get the scores
    documents = d_f.get_list()

    contents_list = []

    tfidf_scores = {}

    # Applying the TF*IDF function to rank the top 20 terms in the documents (that have greater than 20 terms) 
    for doc in documents:
        if len(doc.terms) > 20:
            tfidf_score = my_tfidf(doc, d_f, d_f.get_list_size())
            # Ensure the scores are in descending order
            tfidf_score = dict(sorted(tfidf_score.items(), key=lambda item: item[1], reverse=True))

            tfidf_scores[doc.newsid] = tfidf_score
            # Get the top 20 terms
            top_20_terms = list(tfidf_score.items())[:20]

            print(f'The top 20 terms for document {doc.newsid} are:')
            contents_list.append(f'The top 20 terms for document {doc.newsid} are:\n')
            for term, score in top_20_terms:
                print(f'{term} : {score}')
                contents_list.append(f'{term} : {score}\n')

            print('----------------------------------')
            contents_list.append('----------------------------------\n')
            print('----------------------------------')
            contents_list.append('----------------------------------\n')
    
    for query in Q:
        print('\nThe Ranking Result for query: ', query)
        contents_list.append(f'\nThe Ranking Result for query: {query}\n\n')

        query = parse_q(query, 'common-english-words.txt')

        # Ranking function = sum of the term's frequency in the document * the term's tfidf score
        ranking_scores = {}

        
        for doc in documents:

            ranking_score = 0

            for term in query:
                if term in doc.terms:
                    tfidf_score = tfidf_scores[doc.newsid][term]
                    query_freq = query[term]

                    ranking_score += doc.terms[term] * tfidf_score * query_freq
            
            ranking_scores[doc.newsid] = ranking_score

        
        ranking_scores = dict(sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True))

        for doc in ranking_scores:
            contents_list.append(f'{doc} : {ranking_scores[doc]}\n')
            print(f'{doc} : {ranking_scores[doc]}')
        
    print('----------------------------------')
    contents_list.append('----------------------------------\n')
        
    write_task_file(contents_list, 'Q2')


main_function_q2(d_f, test_queries)


# # Question 3: BM25-based IR Model
# 
# BM25 is a popular IR model with an effective ranking algorithm

# You can use the `DocV3` collection to calculate these variables, such as `N` and `n_i` (you may assume `R=r_i=0`)


# ## Task 3.1
# Define a Python function `avg_length(coll)` to calculate and return the average document length of all documents in the collection `coll`
# - In the `DocV3` class, for the variable (attribute) `doc_size` (the document length), add accessor (get) and mutator (set) methods for it.
# - You may modify your code defined in Question 1 by calling the mutator method of `doc_size` to save the document length in a `DocV3` object when creating the `DocV3` object. At the same time, sum up every `DocV3`'s  `doc_size` as `totalDocLength`, the at the end, calculate the average document length and return it


def avg_length(coll):
    
    # Get the List Doc object length
    coll_length = coll.get_list_size()

    sum_doc_length = 0

    for doc in coll.get_list():
        doc_size = doc.get_doc_size()
        sum_doc_length += doc_size
    
    avg_doc_length = int(sum_doc_length / coll_length)

    return avg_doc_length

avg_length(d_f)


# ## Task 3.2

## Remember
# n_i = number of documents containing a term in the query
# N = number of documents in the collection (found from get_doc_size)
# f_i = frequency weight of the term in the document 
# qf_i = frequency weight of the term in the query
df_bm25 = parse_docs('common-english-words.txt', document_folder)

query = 'US EPA ranks Geo Metro car most fuel-efficient 1997 car'
def my_bm25(coll, q, df):
    
    N = coll.get_list_size()
    R = 0
    r_i = 0
    k_1 = 1.2
    k_2 = 500
    b = 0.75

    contents_list = []

    bm25_rankings = {}

    avg_doc_length = avg_length(coll)    

    parsed_query = parse_q(q, 'common-english-words.txt')

    for doc in coll.get_list():        
        docid = doc.get_news_id()
    
        doc_length = doc.get_doc_size()
        bm25_ranking = 0
        K = k_1 * ((1 - b) + b * (doc_length / avg_doc_length)) 
                
        n_i = 0
        
        for term, count in doc.get_terms_list().items():
            
            if term in parsed_query:

                n_i = count
                f_i = count
                qf_i = parsed_query[term]
                term_bm25_ranking = np.log(((r_i + 0.5) / (R - r_i + 0.5)) / ((n_i - r_i + 0.5) / (N - n_i - R + r_i + 0.5))) * (((k_1 + 1) * f_i) / (K + f_i)) * (((k_2 + 1) * qf_i)/ (k_2 + qf_i))
                bm25_ranking += term_bm25_ranking

        bm25_rankings[docid] = bm25_ranking

        # print(f'Document ID: {docid}, Doc Length: {doc_length} -- BM25 Score: {bm25_ranking}')
        contents_list.append(f'Document ID: {docid}, Doc Length: {doc_length} -- BM25 Score: {bm25_ranking}\n')
        
    # Sorting the top 5 bm25 rankings
    # bm25_rankings = dict(sorted(bm25_rankings.items(), key=lambda item: item[1], reverse=True)[:5])

    # BM25 rankings for the whole collection
    bm25_rankings = dict(sorted(bm25_rankings.items(), key=lambda item: item[1], reverse=True))


    # print(f'\n\nFor Query "{q}", the top-5 relevant documents are:')
    contents_list.append(f'\n\nFor Query "{q}", the top-5 relevant documents are:\n')

    for docid, score in bm25_rankings.items():
        contents_list.append(f'Document ID: {docid}, BM25 Score: {score}\n')
        # print(f'Document ID: {docid}, BM25 Score: {score}')

    return bm25_rankings
    # write_task_file(contents_list, 'Q3')

my_bm25(d_f, q = query, df = df_bm25)



# ## Task 3.3 
# Define a main function to implement a BM25-based IR model to rank documents in the given document collection `RCV1v3` using your functions.
# - You are required to test all the following queries:
#     - The British-Fashion Awards
#     - Rocket attacks
#     - Broadcast Fashion Awards
#     - US EPA ranks Geo Metro car most fuel-efficient 1997 car
# - The BM25-based IR model needs to print out the ranking result (sort them by using each document's ranking score in descending order) of top-5 possible relevant documents for the given query and append outputs into the text file ("your full name_Q3.txt").


query_list = [
    'The British-Fashion Awards',
    'Rocket attacks',
    'Broadcast Fashion Awards', 
    'US EPA ranks Geo Metro car most fuel-efficient 1997 car'
]

def main_function_q3(coll, q):
    
    contents_list = []


    avg_doc_length = avg_length(coll)
    print(f'Average Document Length for this collection is: {avg_doc_length}')

    contents_list.append(f'Average Document Length for this collection is: {avg_doc_length}\n')

    for query in q:
        print(f'The query is: {query}\n\n')
        contents_list.append(f'The query is: {query}\n\n')

        bm25_score = my_bm25(coll, query, df_bm25)

        for doc in coll.get_list():
            docid = doc.get_news_id()
            print(f'Document ID: {docid}, Doc Length: {doc.get_doc_size()} -- BM25 Score: {bm25_score[docid]}')
            contents_list.append(f'Document ID: {docid}, Doc Length: {doc.get_doc_size()} -- BM25 Score: {bm25_score[docid]}\n')
        print('----------------------------------')
        contents_list.append('----------------------------------\n')

        print(f'For query "{query}", the top-5 relevant documents are:')
        contents_list.append(f'For query "{query}", the top-5 relevant documents are:\n')

        for docid, score in list(bm25_score.items())[:5]:
            print(f'Document ID: {docid}, BM25 Score: {score}')
            contents_list.append(f'Document ID: {docid}, BM25 Score: {score}\n')
        print('----------------------------------')
        contents_list.append('----------------------------------\n')

    write_task_file(contents_list, 'Q3')
                

    return 


main_function_q3(d_f, query_list)


