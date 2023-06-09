import os
import nltk 
from nltk.tokenize import word_tokenize
import pdfplumber
from nltk.stem import PorterStemmer
#from docx import Document
from nltk.corpus import stopwords

#-----------tokenizer--------------------------------------------------------------------


def extract_file(folder_path):
    files =[]
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        files.append(file_path)
    return (files)
def get_file_type(path):
    _, file_extention = os.path.splitext(path)
    if file_extention == '.txt':
        return 'text'
    elif file_extention == '.docx':
        return 'word'
    elif file_extention == '.pdf':
        return 'pdf'
    else :
        return 'unsupported file'
def tokenizer(files):
    type = get_file_type(files)
    if type== 'text':
        with open(files ,'r') as file:
            text = file.read()
        tokens = word_tokenize(text)
        return tokens
    # elif type == 'word':
    #     doc = Document(files)
    #     text = ' '.join([paragraph.text for paragraph in doc.paragraphs ])
    #     tokens = word_tokenize(text)
    #     return tokens
    elif type == 'pdf':
        with pdfplumber.open(files) as pdf:
            text = ""
            for page in pdf.pages:
                text+=page.extract_text()
            
        tokens = word_tokenize(text)
        return tokens
    else :
        return '!!!  invalid file type please enter only text | pdf | word files'
    
def tokenize(folder_path):
    files = extract_file(folder_path)
    token_list = []
    for file in files:
        token_list.append(tokenizer(file))
    return token_list

    
# ---------------------stemmer----------------------------


def stemm(folder_path):

    # before stemmming first we have to tokenizw the words
    token_list = tokenize(folder_path)

    # the token list is to dimentional list
    stemmer = PorterStemmer()

    stemmed_word_list=[]
    for tokens in token_list:
        stemmed_words =[]
        for word in tokens:
            stemmed_word = stemmer.stem(word)
            stemmed_words.append(stemmed_word)
        stemmed_word_list.append(stemmed_words)
    return stemmed_word_list


#-------------------------stopword removal---------------------------
def stopword_removal(folder_path):
    # we used the out put of the stemmint 
    # nltk.download('punkt')
    # nltk.download('stopwords')
    sw_list = stemm(folder_path)
    stop_words = set(stopwords.words('english'))
    result = []
    for lists in sw_list:
        lis = []
        for i in lists:
            if i.casefold() not in stop_words and len(i)>1:
                lis.append(i)
        result.append(lis)
    return result
x = stopword_removal('docs')
for i in x:
    for j in i:
        print(j)



