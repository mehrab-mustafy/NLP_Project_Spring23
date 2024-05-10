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
- syntactic_wellformedness.py
     - get_wellformedness (returns the c3 score after parsing the document with a syntactic parser and then checking the grammar(pos_tags) for it)
     - get_c3_mapped (returns the c3 score from 1-5 to the get_wellformedness function)
- assemble_data.py
     - assembles and prepares a csv file for the classification task, collects the scores for all the essays to train them (might take a little long since it calculates all of the scores and stores it in a csv called - 'data.csv')
- classification.py
     - uses the 3 machine learning algorithms (MLP, Logistic Regression, Naive Bayes) to provide accuracy, f1 score, precision, recall
- embeddings_d.py
     - used to calculate the d1, d2 parameters
     - major functions: cosine_similarity_prompt_essay , get_d2



Packages used:
- Pandas
- NLTK
- Spacy
- spellchecker
- os
- sys
- pkl
- sklearn (for ML models - grad students)
- re
- stanza (used to get the syntactic parsed tree of a document, uses the same backend as CoreNLP from NLTK, Stanza allows users to access the Stanford Java toolkit, Stanford CoreNLP, via its server interface, by writing native Python code. Stanza does this by first launching a Stanford CoreNLP server in a background process, and then sending annotation requests to this server process.)


How to run the project:
1. Navigate to the project directory '~/Project/'. The directory should look something like this:
----Project
    --index.csv
    --run_project.py
    --sample_code_1.py
    --sample_code_2.py
    --syntactic_wellformedness.py
    --embeddings_d.py
    --data_assemble.py
    --classification.py
    --README.txt
    --essays
      --....(100 essay files in txt format)...
    

2. From the command prompt, write: python run_project.py 'path_to_your_essay_file.txt'
   (e.g. python run_project.py essays/38209.txt )
   (the path should not use quotes, use relative path to the file from the base project directory)
   (the program finds the prompts from the index.csv file based on the file name provided, hence no need to provide prompts separately, please make sure there is an entry for the prompt of the essay in index.csv file)

3. Additional steps to run for the Machine Learning Classification:
     - From the command prompt, write: python data_assemble.py (this creates the csv file containing scores of the essays that is later fed to the model, the generated csv file has been provided with the project, named - 'data.csv', but if you want you can run the command to generate it on your own)
     - From the command prompt, write: python classification.py (this runs the 3 ML models and provides the accuracy, f1score, precision, recall in the terminal)


Additional Information about Model Training:
     - The data was split into 80-20 split, having 80% of data as the training data and 20% of data as test data.
     - The features are the 2*a, -1*b, c1, c2, 2*c3, 3*d1, d2 - which are the components of the equation to calculate high and low essays 


Sample Output for the scoring with the help of the equation:
	a:2, b:1, c1:5, c2:1, c3: 2, d1: 4, d2: 4.0
	The final score of this essay is: 29.0
	This essay is classified as: low

Sample Output for the Machine Learning Models:
    MLP:
	Accuracy: 0.95, Precision: 1.0, Recall: 0.8888888888888888, F1 Score: 0.9411764705882353

    Logistic Regression:
	Accuracy: 1.0, Precision: 1.0, Recall: 1.0, F1 Score: 1.0

    Naive Bayes
	Accuracy: 0.95, Precision: 1.0, Recall: 0.8888888888888888, F1 Score: 0.9411764705882353

Special Note:
	Please keep the 'w2v.pkl' file in the base directory of the project. This file is used to find the embeddings of the sentences. Because of its large size the upload could not be staged. 