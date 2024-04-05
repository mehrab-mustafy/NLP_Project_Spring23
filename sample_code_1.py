import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words
nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
import string

def num_sentences(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    pos_tags_list = [tag[1] for tag in pos_tags]
    return pos_tags_list



def remove_punctuation(sentence):
    sentence_without_punct = ""
    for char in sentence:
        if char not in string.punctuation:
            sentence_without_punct += char
    return sentence_without_punct

def spelling_mistakes(text):
    text_without_punct = remove_punctuation(text)
    tokens = word_tokenize(text_without_punct)
    english_vocab = set(words.words())
    misspelled_words = [word for word in tokens if word.lower() not in english_vocab]
    print(f"misspelled words: {misspelled_words}")
    return len(misspelled_words)
