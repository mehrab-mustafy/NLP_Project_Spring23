import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
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
    # df['num_mistakes']  = df['file_contents'].apply(spelling_mistakes)    
    # max_mistakes, min_mistakes = df['num_mistakes'].max(), df['num_mistakes'].min()
    # print(df.columns)
    # print(max_mistakes, min_mistakes)
    
    # high_grade_df = df[df['grade'] == 'high']
    # high_grade_df['num_mistakes'] = high_grade_df['file_contents'].apply(spelling_mistakes)
    # max_mistakes1, min_mistakes1 = high_grade_df['num_mistakes'].max(), high_grade_df['num_mistakes'].min()

    
    # low_grade_df = df[df['grade'] == 'low']
    # low_grade_df['num_mistakes'] = low_grade_df['file_contents'].apply(spelling_mistakes)
    # max_mistakes2, min_mistakes2 = low_grade_df['num_mistakes'].max(), low_grade_df['num_mistakes'].min()

    # print(max_mistakes1, min_mistakes1)
    # print(max_mistakes2, min_mistakes2)
    
    text = df.loc[0, 'file_contents']
    text = 'I am writing I am well'
    print(text)
    a = num_sentences(text)
    print(a)
    b = spelling_mistakes(text)
    print(b)
    # s1 = 'Jessica is 8 years old'
    # s2 = 'The goats are eating grass'
    # tags1 = num_sentences(s1)
    # tags2 = num_sentences(s2)
    # print(s1, tags1)
    # print(s2, tags2)
    # c_1, c_2, c_3 = agreement()
    # d_1, d_2 = verbs()
    # print(a, b)
    # Final_Score = 2*a - b + c_1 + c_2 + 2*c_3 + 3*d_1 + 2*d_2
    # print(f"The final score of this essay is: {Final_Score}")
    # print("This essay is classified as: high")
    # print("This essay is classified as: low")
    return 0



if __name__ == '__main__':
    exit(main())