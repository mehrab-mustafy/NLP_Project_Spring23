Teammate 1 NAME: Mehrab Mustafy Rahman
netid: mrahm

Teammate 2 NAME: Mohammad Anas Jawad
netid: mjawad4

Project files & Functions:
- run_project.py: 
     - the main file of the project, run this file to start the project.
     - read_file_contents
- sample_code_1.py: 
     - num_sentences function calculates scoring criterion a (number of sentences and lengths)
     - spelling_mistakes function calculates scoring criterion b (spelling mistakes)
- sample_code_2.py:
     - agreement function calculates scoring criterion c.i (agreement within the sentence)
     - verbs function calculates scoring criterion c.ii (verb mistakes)
     - sentence_splitter
     - tags_after_token_preprocessing
     - subject_verb_agreement_noun
     - subject_verb_agreement_pronoun
     - verb_tense_agreement



Packages used:
- Pandas
- NLTK
- Spacy
- spellchecker
- os
- sys

How to run the project:
1. Navigate to the project directory '~/Project/'. The directory should look something like this:
----Project
    --index.csv
    --run_project.py
    --sample_code_1.py
    --sample_code_2.py
    --README.txt
    --essays
      --....(100 essay files in txt format)...

2. From the command prompt, write: python run_project.py 'path_to_your_essay_file.txt'
   (the path should not use quotes)

