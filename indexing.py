import os
import nltk
from nltk.tokenize import word_tokenize
import pdfplumber
from nltk.stem import PorterStemmer
#from docx import Document
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# --------------this function is defined to extarct document from folder containing the document collection
def extract_file(folder_path):
    files = []
    file_dictionary = {}
    file_no = 1
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        file_dictionary[file_no] = file_path
        files.append(file_path)
        file_no += 1
    return (files, file_dictionary)
# -   ------as the program is designed to operate on multiple documents this function determines the file type of each doc
def get_file_type(path):
    _, file_extention = os.path.splitext(path)
    if file_extention == '.txt':
        return 'text'
    elif file_extention == '.pdf':
        return 'pdf'
    else:
        return 'unsupported file'
# ----this function tokeizes a document------------------------
def tokenizer(files):
    type = get_file_type(files)
    if type == 'text':
        with open(files, 'r') as file:
            text = file.read()
        tokens = word_tokenize(text)
        return tokens
    elif type == 'pdf':
        with pdfplumber.open(files) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        tokens = word_tokenize(text) + [str(files)]
        return tokens
    else:
        return '!!!  invalid file type please enter only text | pdf | word files'
def tokenize(files):
    token_list = []
    for file in files:

        token_list.append(tokenizer(file))
    return token_list
# ---------------------stemmer----------------------------
def stemm(files):

    # before stemmming first we have to tokenizw the words
    token_list = tokenize(files)

    # the token list is two dimentional list
    stemmer = PorterStemmer()

    stemmed_word_list = []
    for tokens in token_list:
        stemmed_words = []
        for word in tokens:
            stemmed_word = stemmer.stem(word)
            stemmed_words.append(stemmed_word)
        stemmed_word_list.append(stemmed_words)
    return stemmed_word_list
# -------------------------stopword removal---------------------------
def stopword_removal(files):
    # we used the out put of the stemmint
    # nltk.download('punkt')
    # nltk.download('stopwords')

    sw_list = stemm(files)
    stop_words = set(stopwords.words('english'))
    result = []
    l = ["''", "``", "'s"]
    for lists in sw_list:
        lis = []
        for i in lists:
            if (i.casefold() not in stop_words) and (len(i) > 1) and (i.casefold() not in l):
                lis.append(i)
        result.append(lis)
    # print(result)
    return result
def vocabulary_file(folder_path):
    files, file_dictionary = extract_file(folder_path)
    final_index_list = stopword_removal(files)
    invertd_term_list = []
    doc_id = 1
    for i in final_index_list:
        for j in i:
            freq = i.count(j)
            invertd_term_list.append(j + " " + str(doc_id) + " " + str(freq))
        doc_id += 1
    result = list(set(invertd_term_list))
    return(sorted(result), file_dictionary)
def find_character_location(file_path, character):
    with open(file_path, 'r') as file:
        content = file.read()
        index = content.find(character)  # Using the 'find()' function
        return index
def searcher(file_path, character):
    with open(file_path, 'r') as file:
        content = file.read()

        if character in content.lower():
            return 1
        else:
            return 0  # Using the 'find()' function
# this block writes the vocabulary file --------------------------------------------
x, y = vocabulary_file('docs/')
file = open('inverted.txt', 'w')
file.write(("{:<15} {:<8} {:<8} {:<15}".format(
    'indecx_term', 'DOC#', 'Freq', 'location'))+'\n\n')
for i in x:
    z = i.split()
    location = find_character_location(y[int(z[1])], z[0])
    content = "{:<15} {:<8} {:<8} {:<15}".format(
        i.split()[0], i.split()[1], i.split()[2], location)
    file.write(content+"\n")
def optimize_query(term_list):
    result = []
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    for i in term_list:
        if (i.casefold() not in stop_words) and (len(i) > 1):
            result.append(stemmer.stem(i))
    return result
file.close()
#  ---------------------------constructiong vector space model------------------------------------------------------------------------------
query = input().strip()
optimized_query = optimize_query(query.split())
print(optimized_query)
file = open('vectorspace.txt', 'w')
file.write(("{:<15} {:<3} {:<3} ".format(
    'indecx_term', 'D1', 'D2'))+'\n\n')
for term in optimized_query:
    l = []
    for i in range(len(y)):
        truth = searcher(y[i+1], term)
        l.append(truth)
    file.write("{:<15} {:<3} {:<3} ".format(
        term, l[0], l[1]))
    file.write('\n')
# --------------------------TF-IDF-------------------------------------
file.close()
#frequency finder ---------------------------------
def freq_finder(file_path, character):
    with open(file_path, 'r') as file:
        content = file.read()

        if character in content.lower():
            frequency = content.lower().count(character)
            return frequency
        else:
            return 0
char_ferq =[]
for i in range(len(y)):
    CharacterFrequencyMap = {} 
    for character in optimized_query:
        frequency = freq_finder(y[i+1], character)
        CharacterFrequencyMap [character] = frequency
    char_ferq.append(CharacterFrequencyMap)
for i in char_ferq:
    print (i)

nltk.download('stopwords')
nltk.download('punkt')

def preprocess(documents):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    preprocessed_documents = []
    for document in documents:
        # Tokenize the document
        tokens = word_tokenize(document.lower())
        # Remove stop words and perform stemming
        filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
        # Join the tokens back into a string
        preprocessed_document = " ".join(filtered_tokens)
        preprocessed_documents.append(preprocessed_document)
    return preprocessed_documents

def build_vsm(documents):
    vectorizer = TfidfVectorizer()
    vsm = vectorizer.fit_transform(documents)
    return vsm, vectorizer.get_feature_names()
