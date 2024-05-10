import pandas as pd
import os
import nltk
import sys
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import words
nltk.download('punkt')
nltk.download('words')
from sample_code_1 import num_sentences, spelling_mistakes
from sample_code_2 import agreement, verbs
from syntactic_wellformedness import get_wellformedness
from embeddings_d import cosine_similarity_prompt_essay, get_d2


def read_file_contents(filename, directory):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        return file.read()
    

def main():
    
    df = pd.read_csv('index.csv', delimiter=';')
    directory = 'essays/'        
    df['file_contents'] = df.apply(lambda row: read_file_contents(row['filename'], directory), axis=1)
    df['file_contents'] = df['file_contents'].str.replace('\n', '').str.replace('\t', '').str.replace("'", '')
    df['file_contents'] = df['file_contents'].str.replace(r'\s+', ' ', regex=True)
    # df['sent_count']  = df['file_contents'].apply(num_sentences)    
    # max_mistakes, min_mistakes = df['num_mistakes'].max(), df['num_mistakes'].min()
    # print(df.columns)
    # print(max_mistakes, min_mistakes)
    
    # high_grade_df = df[df['grade'] == 'high']
    # high_grade_df['sent_count'] = high_grade_df['file_contents'].apply(verbs)
    # max_sent1, min_sent1 = high_grade_df['sent_count'].max(), high_grade_df['sent_count'].min()
            
    # low_grade_df = df[df['grade'] == 'low']
    # low_grade_df['sent_count'] = low_grade_df['file_contents'].apply(verbs)
    # max_sent2, min_sent2 = low_grade_df['sent_count'].max(), low_grade_df['sent_count'].min()

    # print(max_sent1, min_sent1, (max_sent1+min_sent1)/2)
    # print(max_sent2, min_sent2, (max_sent2+min_sent2)/2)
    
    if len(sys.argv) != 2:
        return
    file_path = sys.argv[1]
    try:
        with open(file_path, 'r') as file:
            # Read the contents of the file
            content = file.read()
    except FileNotFoundError:
        print("File not found:", file_path)
    except Exception as e:
        print("An error occurred:", e)

    essay_directory = file_path
    essay_index_name = essay_directory.replace("essays/",'')
    prompt = ""
    df = pd.read_csv("index.csv", sep=";", index_col=False)
    for i in range(len(df)):
        if df.at[i,'filename']==essay_index_name:
            prompt = df.at[i,'prompt']
        
    a = num_sentences(content)
    b = spelling_mistakes(content)
    c_1 = agreement(content)
    c_2 = verbs(content)
    
    c_3 = get_wellformedness(content)
    d_1 = cosine_similarity_prompt_essay(prompt=prompt, essay=content)
    d_2 = get_d2(content)
    
    print(f"a:{a}, b:{b}, c1:{c_1}, c2:{c_2}, c3: {c_3}, d1: {d_1}, d2: {d_2}")
    Final_Score = 2*a - b + c_1 + c_2 + 2*c_3 + 3*d_1 + d_2
    print(f"The final score of this essay is: {Final_Score}")
    if (Final_Score >= 38):
        print("This essay is classified as: high")
    else:
        print("This essay is classified as: low")

    # df2 = pd.read_csv('data.csv', sep=',')
    # map = ''    

    # for i in range(len(df2)):
    #     a = df2.at[i, 'sentence_count']
    #     b = df2.at[i, 'spelling_mistakes']
    #     c1 = df2.at[i, 'agreement']
    #     c2 = df2.at[i, 'verbs']
    #     c3 = df2.at[i, 'c3']
    #     d1 = df2.at[i, 'd1']
    #     d2 = df2.at[i, 'd2']
    #     Score = 2*a - b + c_1 + c_2 + 2*c_3 + 3*d_1 + d_2
    #     if (Score >= 38):
    #         map = 'high'
    #     else:
    #         map = 'low'
    #     df2.at[i, 'predicted'] = map
    # count = 0
    # for i in range(len(df2)):
    #     if(df2.at[i, 'grade'] == df2.at[i, 'predicted']):
    #         count += 1
    
    # accuracy = count/len(df2)
    # print(accuracy)
    # TP = count
    # FP = len(df2) - count  # False Positives: Incorrect predictions
    # TN = 0  # True Negatives: Not applicable if you're calculating metrics for a binary classification problem
    # FN = 0  # False Negatives: Not applicable if you're calculating metrics for a binary classification problem

    # # Calculate Precision
    # precision = TP / (TP + FP)

    # # Calculate Recall
    # recall = TP / (TP + FN)

    # # Calculate F1 Score
    # f1_score = 2 * (precision * recall) / (precision + recall)

    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    # print(f'F1 Score: {f1_score}')
    
    return 0

# 599905.txt 609558.txt

if __name__ == '__main__':
    exit(main())
