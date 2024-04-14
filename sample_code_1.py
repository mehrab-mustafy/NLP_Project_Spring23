import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import words
nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
import spacy
from spellchecker import SpellChecker
nlp = spacy.load("en_core_web_sm")



def num_sentences(text):
    text = 'I eating a sandwich'
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    pos_tags_list = [tag[1] for tag in pos_tags]
    
    
    return pos_tags_list


spell = SpellChecker()
def spelling_mistakes(text):
    doc = nlp(text)
    misspelled_words = []
    for token in doc:
        if not token.is_punct and not token.is_stop:
            if token.text.lower() not in spell:
                misspelled_words.append(token.text)
    # print(f"Misspelled words: {misspelled_words}")
    # print(f"Number of misspelled words: {len(misspelled_words)}")
    if(len(misspelled_words) <= 5):
        return 0
    elif(len(misspelled_words) > 5 and len(misspelled_words) <= 15):
        return 1
    elif(len(misspelled_words) > 15 and len(misspelled_words) <= 25):
        return 2
    elif(len(misspelled_words) > 25 and len(misspelled_words) <= 40):
        return 3
    else:
        return 4 