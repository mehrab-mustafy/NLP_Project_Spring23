import numpy as np
import pickle as pkl
import nltk

# Before running code that makes use of Word2Vec, you will need to download the provided w2v.pkl file
# which contains the pre-trained word2vec representations
#
# If you store the downloaded .pkl file in the same directory as this Python
# file, leave the global EMBEDDING_FILE variable below as is.  If you store the
# file elsewhere, you will need to update the file path accordingly.
EMBEDDING_FILE = "w2v.pkl"

# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)

# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    # Initialize NLTK tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK tokenizer not found, downloading...")
        nltk.download('punkt')
    return nltk.tokenize.word_tokenize(inp_str)

# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300, )

    # [Write your code here:]
    if token in word2vec:
        word_vector = word2vec[token]

    return word_vector

# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function embeds the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, sentence):
    embedding = np.zeros(300, )

    # Write your code here:
    token_list = get_tokens(sentence)
    for token in token_list:
        token_embedding = w2v(word2vec, token)
        embedding += token_embedding
    embedding /= len(token_list)

    return embedding

# Function to get cosine similarity see Equation 6.10 (Jurafsky & Martin v3) for reference
# Arguments:
# a: A numpy vector of size (x, )
# b: A numpy vector of size (x, )
# Returns: sim (float)
# Where, sim (float) is the cosine similarity between vectors a and b. x is the size of the numpy vector. Assume that both vectors are of the same size.
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    sim = 0.0

    # [Write your code here:]
    dot_product = np.dot(a,b)
    mag_a = np.sqrt(np.sum(a**2))
    mag_b = np.sqrt(np.sum(b**2))
    if mag_a!=0 and mag_b!=0:
        sim = dot_product/(mag_a * mag_b)

    return sim

# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. Some of the provided sample code will help you in answering
# questions, but it won't work correctly until all functions have been implemented.
if __name__ == "__main__":
    # Load the Word2Vec representations so that you can make use of it later
    print("Loading Word2Vec representations....")
    word2vec = load_w2v(EMBEDDING_FILE)

    # Example cosine similarity calculation for string to vec
    string1 = "Trump has said that he designated the materials he took to Mar-a-Lago as personal records while still in office"
    string2 = "There were also reports of tornadoes and damage in Mercer and Logan counties in Ohio, according to the weather service"
    string3 = "Smith was in court for the hearing, as well, and Trump eyed him during a break in the proceedings and again when they concluded"

    s2v_string1 = string2vec(word2vec, string1)
    s2v_string2 = string2vec(word2vec, string2)
    s2v_string3 = string2vec(word2vec, string3)

    sim_1 = cosine_similarity(s2v_string1, s2v_string2)
    print("Cosine similarity for String 1 and String 2: {0:.2}".format(sim_1))
    sim_2 = cosine_similarity(s2v_string2, s2v_string3)
    print("Cosine similarity for String 2 and String 3: {0:.2}".format(sim_2))
    sim_3 = cosine_similarity(s2v_string1, s2v_string3)
    print("Cosine similarity for String 1 and String 3: {0:.2}".format(sim_3))

