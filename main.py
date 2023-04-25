### Imports
import numpy as np  # For linear algebra
import pandas as pd  # For data processing, eg. working with a CSV file
import re  # For working with strings
import os # For creating new files to store documents and vectors
import csv # For handling new files

###############
# Word vector #
###############

def word_vector(string: str) -> pd.DataFrame:
    """This function takes a string (corpus) and converts it into a word vector in form of a dataframe.
    The dataframe contains the words as index and the term frequency as values.

    Parameters
    ----------
    string : str
        The string that should be converted into a word vector.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the words as index and the term frequency as values.

    Examples
    --------
    >>> word_vector("This is a test string")
            0
    this    1
    is      1
    a       1
    test    1
    string  1"""

    ls_words = [] # Empty list of words
    dic = {} # Empty dictionary

    # Remove every special character and punctuation, as well as lowering all letters
    ls_words = re.split("\s+",re.sub("\W+"," ",string).lower())

    # Create dictionary for every word
    # Term frequency can be determined
    for word in ls_words:
        if word in dic:
            dic[word] += 1
        else:
            dic[word]=1

    return pd.DataFrame.from_dict(dic,orient="index")

####################
# Similarity check #
####################

def similarity_check(document_str: str, add_document: bool) -> None:
    """This function takes a string as input and compares it to all other documents in the corpus.
It calculates the euclidean distance between the word vector of the input document and all other documents in the corpus.
The function returns the 3 most similar documents to the input document.

Parameters
----------
document_str : str
    The string which should be compared to previous checked documents.

Returns
-------
    The function returns the 3 most similar documents to the input document."""
    
    # Checks if there existing data files or not
    path = os.getcwd() # Get current working directory, for file creation
    if not os.path.exists(path+"/word_vec.csv"): # If there is not one, create a new one
        with open(path+"/word_vec.csv", 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['words'])
    if not os.path.exists(path+"/doc_by_id.csv"): # If there is not one, create a new one
        with open(path+"/doc_by_id.csv", 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['doc_id','text'])

    # Load in Data
    df_whole_doc = pd.read_csv(path+"/doc_by_id.csv") # CSV file with previous documents (whole corpus)
    df_word_vec = pd.read_csv(path+"/word_vec.csv") # CSV file with previous documents (word vector form)

    # Document processing
    # Add input corpus to whole_doc and assign a doc_id
    doc_id = len(df_word_vec.columns) # This id is used in both csv files to combine the word vector with the whole corpus
    # Make a new entry in csv file which stores the whole corpus, not just the words
    new_entry = pd.DataFrame({"doc_id":doc_id,"text":document_str},index=[0])
    df_whole_doc = pd.concat([df_whole_doc,new_entry])
    # Calculate word vector of input
    df_wvec = word_vector(document_str).reset_index()
    # Rename columns
    df_wvec.columns = ["words",str(doc_id)]
    # Append new word vector to existing dataframe with other word vectors
    df_word_vec = pd.merge(df_word_vec,df_wvec,how="outer",left_on="words", right_on="words")
    # Fill empty cells with 0 (word is not in corpus) 
    df_word_vec = df_word_vec.fillna(0)

    # Save changes 
    df_whole_doc.to_csv(path+"/doc_by_id.csv",index=False)
    df_word_vec.to_csv(path+"/word_vec.csv",index=False)

    # Check if at least one document exist in file
    if doc_id == 1 and add_document == False:
        print("Add at least one more document to compare similarity!")
        return None

    # Create dataframe with freqencies of words (columns are different corpuses)
    df_word_vec_freq = pd.DataFrame()
    for col in df_word_vec:
        if not col == "words":
            df_word_vec_freq[col]= df_word_vec[col]/df_word_vec[col].sum()

    # Word vector (numpy array) of search document
    np_wv_search = df_word_vec_freq[str(doc_id)].to_numpy()

    # Create list for the euclidean distance (every column vs. last column)
    ls_distance = [] # Empty list
    for col in df_word_vec_freq:
        if not col == str(doc_id): # Make sure not to compare to own column
            np_wv_tmp = df_word_vec_freq[col].to_numpy()
            # Euclidean distance
            ls_distance.append(np.linalg.norm(np_wv_search-np_wv_tmp))

    # Sort list by highest similarity
    sort_ls = sorted(range(len(ls_distance)), key=lambda k: ls_distance[k])

    # Print top 3 most similar text with similarity count
    for index, key in enumerate(sort_ls):
        # "add_document" argument is used for adding documents to CSV files, and will not print an output
        if add_document == False:
            print("-"*20)
            print("This is the "+ str(index+1) +". most similar document to the search document, with an euclidean distance of "+ str(ls_distance[key]) + ":")
            print("\"" + df_whole_doc.at[key,"text"] + "\"")

        if index == 2: break # Display max 3 most similar documents 

#########
# Tests #
#########

# Documents for testing
d1 = "AI is our friend 34"
d2 = "This program is testing document similarity, by using the euclidean distance method."
d3 = "AI and humans have always been friendly"
d4 = "Three years later, the coffin was still full of Jello."
d5 = "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go."
d6 = "The person box was packed with jelly many dozens of months later."
d7 = "He found a leprechaun in his walnut shell."
d8 = "He found a leprechaun in his walnut."
documents = [d1, d2, d3, d4, d5, d6, d7, d8]

# Adding documents to CSV files
for doc in documents:
    similarity_check(doc, True)

# Example of calling similarity check, with add_document argument set to False
similarity_check("Foundations of Data Science is the best course at CBS!", False)