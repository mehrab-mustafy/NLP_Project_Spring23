import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import words
nltk.download('punkt')
nltk.download('words')
from sample_code_1 import num_sentences, spelling_mistakes
from sample_code_2 import agreement, verbs


def read_file_contents(filename, directory):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        return file.read()
    

def main():
    
    #this_input = input("Enter the Essay to grade: ")

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
    # high_grade_df['sent_count'] = high_grade_df['file_contents'].apply(num_sentences)
    # max_sent1, min_sent1 = high_grade_df['sent_count'].max(), high_grade_df['sent_count'].min()
    # # filtered_df = high_grade_df[high_grade_df['sent_count'] == 49]

    # # Access the 'file_contents' column of the filtered DataFrame
    # # file_contents_49 = filtered_df['file_contents']
    # # for contents in file_contents_49:
    # #     print(contents)
            
    # low_grade_df = df[df['grade'] == 'low']
    # low_grade_df['sent_count'] = low_grade_df['file_contents'].apply(num_sentences)
    # max_sent2, min_sent2 = low_grade_df['sent_count'].max(), low_grade_df['sent_count'].min()

    # print(max_sent1, min_sent1, (max_sent1+min_sent1)/2)
    # print(max_sent2, min_sent2, (max_sent2+min_sent2)/2)
    
    text = df.loc[65, 'file_contents']
    a = num_sentences(text)
    b = spelling_mistakes(text)
    c_1, c_2, c_3 = agreement(text)
    d_1, d_2 = verbs(text)
    
    print(f"a:{a}, b:{b}, c1:{c_1}, c2:{c_2}, c3: {c_3}, d1: {d_1}, d2: {d_2}")
    Final_Score = 2*a - b + c_1 + c_2 + 2*c_3 + 3*d_1 + 2*d_2
    
    print(f"The final score of this essay is: {Final_Score}")
    # print("This essay is classified as: high")
    # print("This essay is classified as: low")
    return 0



if __name__ == '__main__':
    exit(main())