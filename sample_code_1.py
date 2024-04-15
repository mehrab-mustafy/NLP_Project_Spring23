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
    '''counts the total number of sentences in the essay, first splits using periods
    after that checks for capitalization, then finite verb counts, then checks whether there
    are conjunctions/linkers to give the overall number of sentences and associated score. (details provided in the Report file)
    '''
    sentences = []
    current_sentence = ''
    sentence_delimiters = ['.', '!', '?'] 
    for char in text:
        if char not in sentence_delimiters:
            current_sentence += char
        else:
            current_sentence += char
            sentences.append(current_sentence.strip())
            current_sentence = ''
    if current_sentence:
        sentences.append(current_sentence.strip())
    sent_count = len(sentences)
    
    finite_verbs = ['VBD','VBP','VBZ']
    non_finite_verbs = ['VBG', 'VBN']
    ambiguous_verbs = ['VB']
    conjunctions = ['CC']
    others = ['WDT', 'XX']

    for sentence in sentences:
        finite_verb_count = 0
        non_finite_verb_count = 0
        conjunction_count = 0
        other_count = 0
        uppercase_count = 0
        additional_sent_count = 0
        tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        pos_tags_list = [tag[1] for tag in pos_tags]
        for i in range (len(tokens)):
            word = tokens[i].lower()
            if word in ['that', 'because']:
                pos_tags_list[i] = 'XX'
        words = sentence.split()
        uppercase_count = 0
        for word in words:
            if word[0].isupper() and word != "I":
                pos_tag = nltk.pos_tag([word])[0][1]
                #print(pos_tag)
                if pos_tag not in ['NN', 'NNP', 'NNPS']:
                    uppercase_count += 1
        # print(uppercase_count)
        # print(pos_tags_list)
        for tags in pos_tags_list:
            if tags in finite_verbs:
                finite_verb_count += 1
            if tags in non_finite_verbs:
                non_finite_verb_count += 1
            if tags in conjunctions:
                conjunction_count += 1
            if tags in others:
                other_count += 1
        if uppercase_count > 1:
            sent_count = sent_count + uppercase_count - 1
            # print(sentence, uppercase_count, sent_count, '1')
        else:
            if (finite_verb_count - conjunction_count - other_count == 1):
                additional_sent_count = 0
                sent_count = sent_count + additional_sent_count
                # print(sentence, sent_count, '2')
            elif (finite_verb_count - conjunction_count - other_count > 1):
                # print(finite_verb_count, conjunction_count, other_count)
                additional_sent_count = finite_verb_count - conjunction_count - other_count - 1
                sent_count = sent_count + additional_sent_count
                # print(sentence, sent_count, '3')
            else:
                additional_sent_count = 0
                sent_count = sent_count + additional_sent_count
                # print(sentence, sent_count,'4')   
    
    #print(sentence, sent_count)
    # print(sent_count)
    # print(pos_tags_list)
    if sent_count <= 5:
        return 1
    elif sent_count > 5 and sent_count <= 15:
        return 2
    elif sent_count > 15 and sent_count <= 20:
        return 3 
    elif sent_count > 20 and sent_count <= 30:
        return 4
    else:
        return 5



def spelling_mistakes(text):
    '''
    uses the spellchecker and Spacy packages to get the total number of spelling mistakes and return the scores accordingly 
    '''
    spell = SpellChecker()
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