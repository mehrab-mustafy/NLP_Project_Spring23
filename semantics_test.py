import stanza

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

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
doc = nlp('Give me Sunday’s flights arriving in Las Vegas from New York City')
tree = doc.sentences[0].constituency
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
    if (sentence_type[0] == 'NP' and sentence_type[1] == 'VP'):
        sentence_category = 'declarative'
    elif (sentence_type[0] == 'VP' and len(sentence_type) == 1):
        sentence_category = 'imperative'
    elif ((sentence_type[0] == 'VBZ' or sentence_type[0] == 'VBP' or sentence_type[0] == 'MD') and sentence_type[1] == 'NP'):
        sentence_category = 'yes_no'
    elif ((sentence_type[0] == 'WHNP' or sentence_type[0] == 'WHADVP') and sentence_type[1] == 'SQ'):
        sentence_category = 'wh_question'
    else:
        sentence_category = 'incorrect'
    print(sentence_category)
    
    for children in child.children:
        sent_tag_list = []
        sentence = str(children)
        sentence = sentence.replace('(','')
        for word in sentence.split():
            if word in pos_tags:
                sent_tag_list.append(word)
        print(sent_tag_list)