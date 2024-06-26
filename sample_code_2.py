import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words
nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

#split essay into sentences for every '.'
def sentence_splitter(essay):
    sentences = []
    current_sentence = ''
    sentence_delimiters = ['.', '!', '?'] 
    for char in essay:
        if char not in sentence_delimiters:
            current_sentence += char
        else:
            # current_sentence += char
            sentences.append(current_sentence.strip())
            current_sentence = ''
    if current_sentence:
        sentences.append(current_sentence.strip())
    return sentences

# preprocesses tag list so that nouns linked by a CC are converted to the plural form of the noun
def tags_after_token_preprocessing(tokens):
    tags = nltk.pos_tag(tokens)
    tags = [tag for _, tag in tags]
    for i in range (1,len(tags)-1):
        if tags[i]=='CC' and tokens[i]=='and':
            if tags[i-1] in ['NNP', 'NN', 'PRP'] and tags[i+1]in ['NNP', 'NN', 'PRP']:
                tags[i]='NNPS'
                tags[i-1]=''
                tags[i+1]=''
    return [x for x in tags if x not in ['']]

# defines rule set when subject is NN, NNS, NNP, or NNS and returns boolean value representing whether a subject-verb mismatch has been found
def subject_verb_agreement_noun(subject_tag, corresponding_word, verb_tag):
    # define rules for subject-verb agreement
    tag_rule_set = {}
    tag_rule_set['NN'] = ['VBD', 'VBG', 'VBZ']
    tag_rule_set['NNS'] = ['VB', 'VBD', 'VBG', 'VBP']
    tag_rule_set['NNP'] = ['VBD', 'VBG', 'VBZ']
    tag_rule_set['NNPS'] = ['VB', 'VBD', 'VBG', 'VBP']

    allowed_tags = tag_rule_set.get(subject_tag, '')
    if verb_tag not in allowed_tags:
        # print('Subject verb disagreement')
        # print(f'Given subject tag: {subject_tag}')
        # print(f'Corresponding word: {corresponding_word}')
        # print(f'Given verb tag: {verb_tag}')
        # print(f'Expected verb tag: {allowed_tags}')
        return False
    return True

# defines rule set when subject is PRP and returns boolean value representing whether a subject-verb mismatch has been found
def subject_verb_agreement_pronoun(corresponding_word, verb_tag):
    # define rules for subject-verb agreement
    corresponding_word = corresponding_word.lower()
    allowed_tags = []
    if corresponding_word in ['i', 'you', 'we', 'they']:
        allowed_tags = ['VB', 'VBD', 'VBP']
    elif corresponding_word in ['he', 'she', 'it']:
        allowed_tags = ['VBD', 'VBZ']
    elif corresponding_word in ['him', 'her', 'me', 'them', 'us', 'myself', 'yourself', 'himself', 'herself', 'itself', 'themselves', 'ourselves', 'yourselves']:
    # else:
        allowed_tags = ['VBG', 'VBN']
    if verb_tag not in allowed_tags:
        # print('Subject verb disagreement')
        # print(f'Given word: {corresponding_word}')
        # print(f'Given tag: {verb_tag}')
        # print(f'Expected tag: {allowed_tags}')
        return False
    return True

# checks sequence of tags and uses subject-verb agreement functions above to check for mismatch
# returns a score from 1-5 depending on number of errors found
def agreement(essay):
    # testing subject-verb agreement
    # test_sent = 'I told him that I will visited tomorrow'
    # essay = """
    #     Some people want to try new things while other prefer to do the same that they knew all over their life.  I agree with the statement which says that successful people try to do new things even take risks rathre trying known things.  I agree becuse of severals resons.

    #     The frist reson is that  successful people shoud try to do new things otherwase they will not bring new things.  For example, the Neoten is well- known and pioneeir in this feild.  He looked to the apple which fall down from the tree as other but He looked to this case different from other and asked himself why did it fall.  As a result, he interdouced a new law of movement.
    #     Secondly, successful people take risk to do something new.  Forthat, although they will not get  benefit of all things, they can success in something new. 
    #     Thirdly,  successful people try to do new things with sometime high risk rathre trying known things. Becuse of high risk they can get high benefits and this according to the statment which says high risk gives sometime high rate. This statement which is high risk gives sometime high rate is mainly used in marking.  
    #     Becuse of doing new things and  taking risk I agree with successful people try to do new things even take risks rathre trying known things.  

    # """
    sentences = sentence_splitter(essay)
    error_count = 0
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        pos_tags = tags_after_token_preprocessing(tokens)
        # print(pos_tags)
        for i in range(len(pos_tags)-1):
            if pos_tags[i] in ['NN','NNS', 'NNP', 'NNPS'] and pos_tags[i+1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                subject_tag = pos_tags[i]
                corresponding_word = tokens[i]
                verb_tag = pos_tags[i+1]
                if subject_verb_agreement_noun(subject_tag, corresponding_word, verb_tag)==False:
                    # print(f'Corresponding word: {tokens[i]}')
                    error_count += 1
            if pos_tags[i] in ['PRP'] and pos_tags[i+1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and (i+1)<len(pos_tags):
                corresponding_word = tokens[i]
                verb_tag = pos_tags[i+1]
                if subject_verb_agreement_pronoun(corresponding_word, verb_tag)==False:
                    error_count += 1
    if error_count>=9:
        return 1
    elif error_count>=6 and error_count<=8:
        return 2
    elif error_count>=4 and error_count<=5:
        return 3
    elif error_count>=2 and error_count<=3:
        return 4
    else:
        return 5

# defines rule set when subject is PRP and returns boolean value representing whether a subject-verb mismatch has been found
def verb_tense_agreement(md_word, verb_tag):
    # define rules for subject-tense agreement
    rule_set = {}
    rule_set['can'] = ['VB', 'IN', 'RB']
    rule_set['could'] = ['VB', 'IN', 'RB']
    rule_set['may'] = ['VB', 'IN', 'RB']
    rule_set['might'] = ['VB', 'IN', 'RB']
    rule_set['shall'] = ['VB', 'IN', 'RB']
    rule_set['should'] = ['VB', 'IN', 'RB']
    rule_set['will'] = ['VB', 'IN', 'RB']
    rule_set['would'] = ['VB', 'IN', 'RB']
    rule_set['ought'] = ['VB', 'IN', 'RB']
    rule_set['must'] = ['VB', 'RB']
    rule_set['am'] = ['VBG', 'VBN', 'IN', 'RB']
    rule_set['is'] = ['VBG', 'VBN', 'IN', 'RB']
    rule_set['was'] = ['VBG', 'VBN', 'IN', 'RB']
    rule_set['are'] = ['VBG', 'VBN', 'IN', 'RB']
    rule_set['were'] = ['VBG', 'VBN', 'IN', 'RB']
    rule_set['been'] = ['JJ', 'VBG']
    rule_set['be'] = ['VBN', 'VBG']
    rule_set['to'] = ['VB', 'VBG']
    # rule_set['in'] = ['VBG']

    md_word = md_word.lower()
    allowed_tags = rule_set.get(md_word, '')
    if verb_tag not in allowed_tags:
        # print('Verb-tense disagreement')
        # print(f'Corresponding word: {md_word}')
        # print(f'Given tag: {verb_tag}')
        # print(f'Expected tag: {allowed_tags}')
        return False
    return True

# checks sequence of tags and uses verb-tense agreement function above to check for mismatch
# returns a score from 1-5 depending on number of errors found
def verbs(essay):
    # testing verb-tense agreement
    # test_sent = 'The modern society rappresented the perfect ambient to influenced the minds of all the person.'
    # test_sent = """
    #     Some people want to try new things while other prefer to do the same that they knew all over their life.  I agree with the statement which says that successful people try to do new things even take risks rathre trying known things.  I agree becuse of severals resons.

    #     The frist reson is that  successful people shoud try to do new things otherwase they will not bring new things.  For example, the Neoten is well- known and pioneeir in this feild.  He looked to the apple which fall down from the tree as other but He looked to this case different from other and asked himself why did it fall.  As a result, he interdouced a new law of movement.
    #     Secondly, successful people take risk to do something new.  Forthat, although they will not get  benefit of all things, they can success in something new. 
    #     Thirdly,  successful people try to do new things with sometime high risk rathre trying known things. Becuse of high risk they can get high benefits and this according to the statment which says high risk gives sometime high rate. This statement which is high risk gives sometime high rate is mainly used in marking.  
    #     Becuse of doing new things and  taking risk I agree with successful people try to do new things even take risks rathre trying known things.  

    # """
    sentences = sentence_splitter(essay)
    error_count = 0
    tokens = word_tokenize(essay)
    pos_tags = tags_after_token_preprocessing(tokens)
    # print(pos_tags)
    for i in range(len(pos_tags)-1):
        # print(i)
        if pos_tags[i] in ['MD', 'VBP', 'VBN'] and pos_tags[i+1] in ['RB'] and i+2 < len(pos_tags):
            md_word = tokens[i]
            # print(md_word)
            verb_tag = pos_tags[i+2]
            # print(verb_tag)
            if verb_tense_agreement(md_word, verb_tag)==False:
                error_count += 1
        if pos_tags[i] in ['MD', 'VBP', 'VBN', 'VBD'] and pos_tags[i+1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            md_word = tokens[i]
            # print(md_word)
            verb_tag = pos_tags[i+1]
            # print(verb_tag)
            if verb_tense_agreement(md_word, verb_tag)==False:
                error_count += 1
        if pos_tags[i] in ['VB'] and tokens[i] in ['be'] and pos_tags[i+1] in ['RB'] and pos_tags[i+2] in ['VB'] and i+2 < len(pos_tags):
            md_word = tokens[i]
            # print(md_word)
            verb_tag = pos_tags[i+2]
            # print(verb_tag)
            if verb_tense_agreement(md_word, verb_tag)==False:
                error_count += 1
        if pos_tags[i] in ['TO'] and pos_tags[i+1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            md_word = tokens[i]
            # print(md_word)
            verb_tag = pos_tags[i+1]
            # print(verb_tag)
            if verb_tense_agreement(md_word, verb_tag)==False:
                error_count += 1
        # if pos_tags[i] in ['IN'] and pos_tags[i+1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        #     md_word = pos_tags[i]
        #     # print(md_word)
        #     verb_tag = pos_tags[i+1]
        #     # print(verb_tag)
        #     if verb_tense_agreement(md_word, verb_tag)==False:
        #         error_count += 1

    # print(f"Error count: {error_count}")


    if error_count>=5:
        return 1
    elif error_count>=4 and error_count<5:
        return 2
    elif error_count>=3 and error_count<4:
        return 3
    elif error_count>=2 and error_count<3:
        return 4
    else:
        return 5
