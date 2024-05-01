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
    "CC": ["NN", "NNS", "NNP", "NNPS", "PRP", "DT", "VB", "RB", "RBR", "JJ", "IN", "VBG", "VBN", "JJR", "JJS", "VBZ", "VBP", "MD", "CD", "WDT", "PRP$", "RP", "TO", "FW"],
    "CD": ["NN", "NNS", "NNP", "NNPS", "JJ", "IN", "DT", "VB", "RB", "VBN", "VBG", "CC", "JJR", "JJS", "VBZ", "VBP", "RP", "TO", "MD", "PRP", "WDT", "PRP$", "FW"],
    "DT": ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "IN", "VBN", "VBG"],
    "EX": ["VBZ"],
    "FW": ["NN", "NNS", "NNP", "NNPS"],
    "IN": ["NN", "NNS", "NNP", "NNPS", "PRP", "DT", "VB", "RB", "JJ", "IN", "VBG", "VBN", "JJR", "JJS", "VBZ", "VBP", "MD", "CD", "WDT", "PRP$", "RP", "TO", "CC", "EX", "PDT", "FW", "WP"],
    "JJ": ["NN", "WRB", "NNS", "NNP", "NNPS", "CC", "IN", "DT", "VB", "RB", "JJ", "JJR", "JJS", "MD", "CD", "WDT", "PRP$", "RP", "TO", "PRP", "EX", "PDT", "FW"],
    "JJR": ["NN", "NNS", "NNP", "NNPS", "IN"],
    "JJS": ["NN", "NNS", "NNP", "NNPS", "IN", "RB"],
    "MD": ["VB", "RB"],
    "NN": ["IN", "DT", "VBD", "WP$", "VB", "RB", "JJ", "VBG", "VBN", "CC", "POS", "JJR", "JJS", "VBZ", "VBP", "MD", "CD", "WDT", "PRP$", "RP", "TO", "PRP", "EX", "PDT", "FW", "WRB"],
    "NNS": ["IN", "DT", "WRB", "VB", "RB", "JJ", "VBD", "VBG", "VBN", "CC", "POS", "JJR", "JJS", "VBZ", "VBP", "MD", "CD", "WDT", "PRP$", "RP", "TO", "PRP", "EX", "PDT", "FW", "WP"],
    "NNP": ["IN", "DT", "VB", "VBD", "RB", "JJ", "NNP", "VBG", "VBN", "CC", "POS", "JJR", "JJS", "VBZ", "VBP", "MD", "CD", "WDT", "PRP$", "RP", "TO", "PRP", "EX", "PDT", "FW"],
    "NNPS": ["IN", "DT", "VB", "RB", "JJ", "VBG", "VBN", "CC", "POS", "JJR", "JJS", "VBZ", "VBP", "MD", "CD", "WDT", "PRP$", "RP", "TO", "PRP", "EX", "PDT", "FW"],
    "PDT": ["DT", "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"],
    "POS": ["NN", "NNS","JJ", "JJS", "VBG"],
    "PRP": ["VB", "VBZ", "VBP", "VBD", "MD", "RB", "TO", "IN", "VBN", "CC"],
    "PRP$": ["NN", "NNS", "NNP", "NNPS", "JJ"],
    "RB": ["VB", "JJ", "RB", "JJR", "VBP", "VBZ", "VBN", "VBD", "IN", "DT", "VBG", "NNP", "TO", "CD", "PRP"],
    "RBR": ["JJ", "RB"],
    "RBS": ["JJ", "RB"],
    "RP": ["VB", "RP", "PRP$", "IN"],
    "SYM": [],
    "TO": ["VB"],
    "UH": [],
    "VB": ["DT", "NN", "NNS", "NNP", "NNPS", "PRP", "RB", "JJ", "IN", "VBG", "VBN", "CC", "JJR", "JJS", "VBZ", "VBP", "MD", "CD", "WDT", "PRP$", "RP", "TO", "EX", "PDT", "FW", "RBR"],
    "VBD": ["RB", "DT", "VB", "TO" "MD", "NN", "NNS", "NNP", "NNPS", "PRP", "PRP$", "VBG", "VBN", "RP", "JJ", "JJR", "JJS", "IN"],
    "VBG": ["VB", "VBN", "WP", "TO", "IN", "NN", "NNS", "NNP", "NNPS", "PRP", "DT", "RB", "JJ", "CC", "MD", "PRP$", "VBZ", "VBP", "CD", "WDT", "RP", "PDT", "FW", "JJ", "JJR", "JJS"],
    "VBN": ["VB", "TO", "IN", "NN", "NNS", "NNP", "NNPS", "PRP", "DT", "RB", "JJ", "CC", "MD", "PRP$", "VBZ", "VBP", "CD", "WDT", "RP", "PDT", "FW", "JJ", "JJR", "JJS"],
    "VBP": ["TO", "WP", "IN", "NN", "NNS", "NNP", "NNPS", "PRP", "DT", "RB", "JJ", "CC", "MD", "PRP$", "VBN", "CD", "WDT", "RP", "PDT", "FW", "JJ", "JJR", "JJS", "WRB", "VBG"],
    "VBZ": ["RB", "VB", "MD", "NN", "NNS", "WRB", "NNP", "NNPS", "PRP", "PRP$", "VBG", "VBN", "RP", "JJ", "JJR", "JJS", "TO", "IN", "DT"],
    "WDT": ["VB", "NN", "NNS", "NNP", "NNPS", "VBZ", "VBP"],
    "WP": ["VB", "VBZ", "VBP", "VBD", "NN", "NNS", "NNP", "NNPS", "PRP", "RB"],
    "WP$": ["VB", "NN", "NNS", "NNP", "NNPS"],
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

def get_wellformedness(doc):
    line_count = 1
    doc = nlp(doc)
    for single_sentence in doc.sentences:
        print("This is for sentence ", line_count)
        print(single_sentence.constituency)
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
            print(sentence_type)
            if (sentence_type[0] == 'NP' and (sentence_type[1] == 'VP' or sentence_type[1] == 'ADVP')):
                sentence_category = 'declarative'
            elif(sentence_type[1] == 'NP' and sentence_type[2] == 'VP'):
                sentence_category = 'declarative'
            elif(sentence_type[0] == 'NP' and sentence_type[-2] == 'VP'):
                sentence_category = 'declarative'
            elif (len(sentence_type) > 2 and sentence_type[2] == 'NP' and (sentence_type[3] == 'VP' or sentence_type[3] == 'ADVP')):
                sentence_category = 'declarative'
            elif (sentence_type[0] == 'S'):
                sentence_category = 'declarative'
            elif (len(sentence_type) > 2 and sentence_type[2] == 'S'):
                sentence_category = 'declarative'   
            elif ((sentence_type[0] == 'VP') and (len(sentence_type) == 1 or sentence_type[-1]=='.')):
                sentence_category = 'imperative'
            elif ((sentence_type[0] == 'VBZ' or sentence_type[0] == 'VBP' or sentence_type[0] == 'MD') and sentence_type[1] == 'NP'):
                sentence_category = 'yes_no'
            elif (len(sentence_type) > 2 and (sentence_type[2] == 'VBZ' or sentence_type[2] == 'VBP' or sentence_type[2] == 'MD') and sentence_type[3] == 'NP'):
                sentence_category = 'yes_no'
            elif ((sentence_type[0] == 'WHNP' or sentence_type[0] == 'WHADVP') and sentence_type[1] == 'SQ'):
                sentence_category = 'wh_question'
            elif(sentence_type[0]=='CC' and sentence_type[2]=='VP'):
                sentence_category = 'complex_sentence'
            elif(sentence_type[0]=='ADVP'):
                for i in range(1, len(sentence_type)):
                    if(sentence_type[i] == 'PP' or sentence_type[i] == 'NP'):
                        sentence_category = 'very complex_sentence'  
            elif(sentence_type[0]=='PP'):
                for i in range(1, len(sentence_type)-1):
                    if(sentence_type[i] == 'NP' or sentence_type[i+1] == 'VP'):
                        sentence_category = 'very complex_sentence'  
            
            else:
                sentence_category = 'incorrect'
            print(sentence_category)
            
            i=0
            for children in child.children:
                print(f"Type of phrase: {sentence_type[i]}")
                i += 1
                sent_tag_list = []
                sentence = str(children)
                sentence = sentence.replace('(','')
                for word in sentence.split():
                    if word in pos_tags:
                        sent_tag_list.append(word)
                print(sent_tag_list)
                error = 0
                for j in range(len(sent_tag_list)-1):
                    current_tag = sent_tag_list[j]
                    following_tag = sent_tag_list[j+1]
                    if following_tag not in tag_followers[current_tag]:
                        print(f"Error tag pair: {current_tag} {following_tag}")
                        error += 1
                print(f"Number of erroneous tags in this phrase: {error}")
                

df = pd.read_csv('index.csv', delimiter=';')
directory = 'essays/'        
df['file_contents'] = df.apply(lambda row: read_file_contents(row['filename'], directory), axis=1)
df['file_contents'] = df['file_contents'].str.replace('\n', '').str.replace('\t', '').str.replace("'", '')
df['file_contents'] = df['file_contents'].str.replace(r'\s+', ' ', regex=True)
file_name = '1174920.txt'
doc = ''
for i in range(len(df)):
    if df.at[i,'filename']==file_name:
        doc = df.at[i, 'file_contents']
print(doc)
get_wellformedness(doc)


# low  : 1007363.txt , 1096747.txt , 1174920.txt , 1181356.txt , 1388870.txt
# high : 1392946.txt , 1827588.txt , 1876159.txt , 279212.txt , 618384.txt