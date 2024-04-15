import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words
nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

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

def subject_verb_agreement_noun(subject_tag, verb_tag):
    # define rules for subject-verb agreement
    tag_rule_set = {}
    tag_rule_set['NN'] = ['VBD', 'VBG', 'VBZ']
    tag_rule_set['NNS'] = ['VB', 'VBD', 'VBG', 'VBP']
    tag_rule_set['NNP'] = ['VBD', 'VBG', 'VBZ']
    tag_rule_set['NNPS'] = ['VB', 'VBD', 'VBG', 'VBP']

    allowed_tags = tag_rule_set.get(subject_tag)
    if verb_tag not in allowed_tags:
        print('Subject verb disagreement')
        print(f'Given subject tag: {subject_tag}')
        print(f'Given verb tag: {verb_tag}')
        print(f'Expected verb tag: {allowed_tags}')
        return False
    return True

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
        print('Subject verb disagreement')
        print(f'Given word: {corresponding_word}')
        print(f'Given tag: {verb_tag}')
        print(f'Expected tag: {allowed_tags}')
        return False
    return True

def agreement(essay):
    essay = """
    This is an important aspect of today time.
    This products rathen are not much better, but today is not important the really character of the product, but only the money and the client not rappresented the important actor in this process.
    Every day any people buy same products that is not rappresented the your necessity, but is only important buy any product.
    To explain this argoment in my nation, at the television, there is an program that discuss of the problem rappresented by this.
    More people go to this program television to talk about your problem, that is very radicate in my nation.
    The modern society rappresented the perfect ambient to influenced the minds of all the person.
    In my self is present the reasons of this statement, that is one of the problem of the life.
    But not all the people and the time is in accord with this problem, because any time the person is too according with the make products.
    Thus I agree with this statement, because this event is present in my life every day, and rappresented the problem with I do fighting.
    But to explain all the aspect about this argoment is very inportant to illustre any examples.
    The television programs that every day introduce in the minds more argoment, news and other problem, or breaking news, is the first actor in this process.
    This opinion rappresented my self in my life, because for me the life of all the people is not possible to influence by the activity of any person.
    The society lose the propriety when this problem will rappresent the must argoment of the talk and the life of the people, because as very difficult live at a time with this argoment.
    The my request is that the new politics discuss about this problem.
    """
    error_count = 0
    tokens = word_tokenize(essay)
    pos_tags = tags_after_token_preprocessing(tokens)
    print(pos_tags)
    for i in range(len(pos_tags)-1):
        if pos_tags[i] in ['NN','NNS', 'NNP', 'NNPS'] and pos_tags[i+1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            subject_tag = pos_tags[i]
            verb_tag = pos_tags[i+1]
            if subject_verb_agreement_noun(subject_tag, verb_tag)==False:
                print(f'Corresponding word: {tokens[i]}')
                error_count += 1
        if pos_tags[i] in ['PRP'] and pos_tags[i+1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and (i+1)<len(pos_tags):
            corresponding_word = tokens[i]
            verb_tag = pos_tags[i+1]
            if subject_verb_agreement_pronoun(corresponding_word, verb_tag)==False:
                error_count += 1
    print(f'Subject-verb error count: {error_count}')
    # if conditions for returning scores from 1-5 depening on number of errors
    return error_count

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
    rule_set['am'] = ['VBG', 'IN', 'RB']
    rule_set['is'] = ['VBG', 'IN', 'RB']
    rule_set['was'] = ['VBG', 'IN', 'RB']
    rule_set['are'] = ['VBG', 'IN', 'RB']
    rule_set['were'] = ['VBG', 'IN', 'RB']
    rule_set['been'] = ['JJ', 'VBG']
    rule_set['be'] = ['VBN', 'VBG']
    rule_set['to'] = ['VB', 'VBG']
    # rule_set['in'] = ['VBG']

    md_word = md_word.lower()
    allowed_tags = rule_set.get(md_word, '')
    if verb_tag not in allowed_tags:
        print('Verb-tense disagreement')
        print(f'Corresponding word: {md_word}')
        print(f'Given tag: {verb_tag}')
        print(f'Expected tag: {allowed_tags}')
        return False
    return True

def verbs(essay):
    # testing verb-tense agreement
    essay = 'The modern society rappresented the perfect ambient to influenced the minds of all the person.'
    # test_sent = """
    # This is an important aspect of today time.
    # This products rathen are not much better, but today is not important the really character of the product, but only the money and the client not rappresented the important actor in this process.
    # Every day any people buy same products that is not rappresented the your necessity, but is only important buy any product.
    # To explain this argoment in my nation, at the television, there is an program that discuss of the problem rappresented by this.
    # More people go to this program television to talk about your problem, that is very radicate in my nation.
    # The modern society rappresented the perfect ambient to influenced the minds of all the person.
    # In my self is present the reasons of this statement, that is one of the problem of the life.
    # But not all the people and the time is in accord with this problem, because any time the person is too according with the make products.
    # Thus I agree with this statement, because this event is present in my life every day, and rappresented the problem with I do fighting.
    # But to explain all the aspect about this argoment is very inportant to illustre any examples.
    # The television programs that every day introduce in the minds more argoment, news and other problem, or breaking news, is the first actor in this process.
    # This opinion rappresented my self in my life, because for me the life of all the people is not possible to influence by the activity of any person.
    # The society lose the propriety when this problem will rappresent the must argoment of the talk and the life of the people, because as very difficult live at a time with this argoment.
    # The my request is that the new politics discuss about this problem.
    # """
    error_count = 0
    tokens = word_tokenize(essay)
    pos_tags = tags_after_token_preprocessing(tokens)
    print(pos_tags)
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

    print(f"Verb-tense error count: {error_count}")

    return error_count
