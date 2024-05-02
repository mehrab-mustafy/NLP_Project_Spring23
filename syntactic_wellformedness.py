import stanza
import pandas as pd
import os

def read_file_contents(filename, directory):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        return file.read()
    
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

# Tag rule set for each of the 36 tags
tag_followers = {
    "CC": ["DT", "JJ", "JJR", "NNS", "NNP", "PRP", "RB", "RP", "TO", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "PRP$", "NN", "RBR"],
    "CD": ["JJ", "NN", "NNS", "NNP", "NNPS"],
    "DT": ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "IN", "VBN", "VBG"],
    "EX": ["VBZ", "MD"],
    "FW": ["NN", "NNS", "NNP", "NNPS"],
    "IN": ["NN", "NNS", "NNP", "NNPS", "PRP", "DT", "VB", "VBG", "VBZ", "VBP", "WDT", "PRP$", "RP", "TO", "EX", "RB", "IN"], # removed JJ
    "JJ": ["NN", "WRB", "NNS", "NNP", "NNPS", "CC", "DT", "VB", "JJ", "WDT", "PRP$", "TO", "PRP", "PDT", "IN"], # removed JJ
    "JJR": ["NN", "NNS", "NNP", "NNPS", "IN"],
    "JJS": ["NN", "NNS", "NNP", "NNPS", "IN", "RB"],
    "MD": ["VB", "RB"],
    "NN": ["IN","VBD", "WP$", "RB", "CC", "POS", "VBZ", "MD", "WDT", "RP", "TO", "EX", "PDT", "WRB", "JJ", "NNS", "DT"],
    "NNS": ["IN","VBD", "WP$", "RB", "CC", "POS", "VBP", "MD", "WDT", "RP", "TO", "EX", "PDT", "WRB", "VBG", "PRP", "WP", "DT"], #added DT
    "NNP": ["IN","VBD", "WP$", "RB", "CC", "POS", "VBZ", "MD", "WDT", "RP", "TO", "EX", "PDT", "WRB", "NNP"],
    "NNPS": ["IN","VBD", "WP$", "RB", "CC", "POS", "VBP", "MD", "WDT", "RP", "TO", "EX", "PDT", "WRB"],
    "PDT": ["DT", "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"],
    "POS": ["NN", "NNS","JJ", "JJS", "VBG"],
    "PRP": ["VB", "VBZ", "VBP", "VBD", "MD", "RB", "TO", "IN", "VBN", "CC"],
    "PRP$": ["NN", "NNS","JJ", "JJR", "JJS", "CD", "CC"],
    "RB": ["VB", "JJ", "RB", "JJR", "VBP", "VBZ", "VBN", "VBD", "IN", "DT", "VBG", "NNP", "TO", "CD", "PRP", "NN", "MD"],
    "RBR": ["JJ", "RB", "IN"], #added IN
    "RBS": ["JJ", "RB"],
    "RP": ["VB", "RP", "PRP$", "IN"],
    "SYM": [],
    "TO": ["VB"],
    "UH": [],
    "VB": ["DT", "NN", "NNS", "PRP", "RB", "JJ", "IN", "VBG", "CC", "JJR", "CD", "WDT", "PRP$", "RP", "TO", "EX", "RBR", "VBN"],
    "VBD": ["RB", "TO", "PRP", "PRP$", "VBG", "VBN", "RP", "JJ", "IN", "DT"],
    "VBG": ["WP", "TO", "IN", "PRP", "DT", "RB", "CC", "PRP$", "CD", "RP", "PDT", "FW", "NNS", "EX", "JJ", "JJR"],
    "VBN": ["TO", "IN", "DT", "RB", "JJ", "CC", "MD", "VBG", "CD", "RP"],
    "VBP": ["RB", "RBR", "NN", "NNS", "WP", "PRP", "PRP$", "VBG", "VBN", "RP", "JJ", "JJR", "TO", "IN", "WRB", "CD", "DT"],
    "VBZ": ["RB", "RBR", "NN", "NNS", "WP", "PRP", "PRP$", "VBG", "VBN", "RP", "JJ", "JJR", "TO", "IN", "DT"],
    "WDT": ["VB", "NN", "NNS", "VBZ", "VBP", "VBD", "JJ"],
    "WP": ["VB", "VBZ", "VBP", "VBD", "NN", "NNS", "PRP", "RB", "MD", "JJ", "CC"],
    "WP$": ["NN", "NNS", "JJ", "CD"],
    "WRB": ["VB", "JJ", "RB", "NN", "PRP", "TO", "NNS", "VBG"]
}

pos_tags = [
    "CC",   # coordinating conjunction
    "CD",   # cardinal digit
    "DT",   # determiner
    "EX",   # existential there
    "FW",   # foreign word
    "IN",   # preposition/subordinating conjunction
    "JJ",   # adjective
    "JJR",  # adjective, comparative
    "JJS",  # adjective, superlative
    "LS",   # list marker
    "MD",   # modal
    "NN",   # noun, singular or mass
    "NNS",  # noun plural
    "NNP",  # proper noun, singular
    "NNPS", # proper noun, plural
    "PDT",  # predeterminer
    "POS",  # possessive ending
    "PRP",  # personal pronoun
    "PRP$", # possessive pronoun
    "RB",   # adverb
    "RBR",  # adverb, comparative
    "RBS",  # adverb, superlative
    "RP",   # particle
    "SYM",  # symbol
    "TO",   # to
    "UH",   # interjection
    "VB",   # verb, base form
    "VBD",  # verb, past tense
    "VBG",  # verb, gerund/present participle
    "VBN",  # verb, past participle
    "VBP",  # verb, sing. present, non-3d
    "VBZ",  # verb, 3rd person sing. present
    "WDT",  # wh-determiner
    "WP",   # wh-pronoun
    "WP$",  # possessive wh-pronoun
    "WRB"   # wh-adverb
]

def get_c3_mapped(raw_score):
    mapped_score = 0
    if raw_score>=1.64:
        mapped_score = 5
    elif raw_score>=1.48 and raw_score<1.64:
        mapped_score = 4
    elif raw_score>=1.32 and raw_score<1.48:
        mapped_score = 3
    elif raw_score>=1.14 and raw_score<1.32:
        mapped_score = 2
    elif raw_score<1.14:
        mapped_score = 1
    return mapped_score

def get_wellformedness(doc):
    # doc = "Give me the ball"
    sentence_score_list = []
    total_correct_sentence_classified = 0
    line_count = 1
    doc = nlp(doc)
    for single_sentence in doc.sentences:
        total_score = 0.0
        # print("**********************************")
        # print("This is for sentence ", line_count)
        # print("**********************************")
        # print(single_sentence.constituency)
        line_count += 1
        tree = single_sentence.constituency
        sent_tag_list = []
        sentence_type = []
        sentence_category = []
        for child in tree.children:
            #sentence type identifier:
            for children in child.children:
                phrase_identifier = str(children)
                phrase_identifier = phrase_identifier.replace('(', '').split()
                sentence_type.append(phrase_identifier[0])

            # Declarative : NP - VP ...
            # Imperative  : VP
            # Yes/No      : VBZ/VBP/MD - NP
            # Wh-Question : WHNP/WHADVP - SQ
            # print(sentence_type)
            if (len(sentence_type) >= 2 and sentence_type[0] == 'NP' and (sentence_type[1] == 'VP' or sentence_type[1] == 'ADVP')):
                sentence_category = 'declarative'
            elif(len(sentence_type) >= 2 and sentence_type[1] == 'NP' and len(sentence_type) > 2 and sentence_type[2] == 'VP'):
                sentence_category = 'declarative'
            elif (len(sentence_type) >= 2 and len(sentence_type) > 3 and sentence_type[2] == 'NP' and (sentence_type[3] == 'VP' or sentence_type[3] == 'ADVP')):
                sentence_category = 'declarative'   
            elif ((sentence_type[0] == 'VP') and (len(sentence_type) == 1 or sentence_type[-1]=='.')):
                sentence_category = 'imperative'
            elif (len(sentence_type) >= 2 and (sentence_type[0] == 'VBZ' or sentence_type[0] == 'VBP' or sentence_type[0] == 'MD') and len(sentence_type) > 1 and sentence_type[1] == 'NP'):
                sentence_category = 'yes_no'
            elif (len(sentence_type) >= 2 and len(sentence_type) > 3 and (sentence_type[2] == 'VBZ' or sentence_type[2] == 'VBP' or sentence_type[2] == 'MD') and sentence_type[3] == 'NP'):
                sentence_category = 'yes_no'
            elif (len(sentence_type) >= 2 and (sentence_type[0] == 'WHNP' or sentence_type[0] == 'WHADVP') and len(sentence_type) > 1 and sentence_type[1] == 'SQ'):
                sentence_category = 'wh_question'
            else:
                sentence_category = 'incorrect'

            # print(sentence_category)
            
            if sentence_category in ['declarative', 'imperative', 'yes_no', 'wh_question']:
                total_correct_sentence_classified += 1

            i=0
            total_tag_pairs = 0 # this stores the total number of tag pairs
            total_error_tags = 0 # this stores the total number of incorrect tag pairs
            for children in child.children:
                # print(f"Type of phrase: {sentence_type[i]}")
                i += 1
                sent_tag_list = []
                sentence = str(children)
                sentence = sentence.replace('(','')
                for word in sentence.split():
                    if word in pos_tags:
                        sent_tag_list.append(word)
                # print(sent_tag_list)
                if len(sent_tag_list)!=0:
                    total_tag_pairs += len(sent_tag_list)-1
                error = 0 # this is for keeping track of the number of incorrect tag pairs under a certain phrase, e.g, under NP or VP
                for j in range(len(sent_tag_list)-1):
                    current_tag = sent_tag_list[j]
                    following_tag = sent_tag_list[j+1]
                    if following_tag not in tag_followers.get(current_tag,''):
                        # print(f"Error tag pair: {current_tag} {following_tag}")
                        error += 1
                total_error_tags += error # updating total number of incorrect tag pairs
                # print(f"Number of erroneous tags in this phrase: {error}")
            if total_tag_pairs!=0:
                # print(f"Total tag pairs: {total_tag_pairs} Total error tag pairs: {total_error_tags}")
                total_score += (total_tag_pairs-total_error_tags)/total_tag_pairs # this gives us a score based on total number of incorrect tag pairs
        # print(f"Score of this sentence: {total_score}")
        sentence_score_list.append(total_score)
    essay_score_raw = sum(sentence_score_list)/len(sentence_score_list) + total_correct_sentence_classified/line_count # this gives us the score of the essay after considering the sentence structures and total number of incorrect tag pairs
    # print(f"Average score of essay: {essay_score}")
    essay_score_c3 = get_c3_mapped(essay_score_raw)
    return essay_score_c3


