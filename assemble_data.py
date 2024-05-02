import pandas as pd
import os
from sample_code_1 import num_sentences, spelling_mistakes
from sample_code_2 import agreement, verbs

def read_file_contents(filename, directory):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        return file.read()
    
df = pd.read_csv('essays_with_c3.csv', delimiter=';')
directory = 'essays/'        
df['file_contents'] = df.apply(lambda row: read_file_contents(row['filename'], directory), axis=1)
df['file_contents'] = df['file_contents'].str.replace('\n', '').str.replace('\t', '').str.replace("'", '')
df['file_contents'] = df['file_contents'].str.replace(r'\s+', ' ', regex=True)
df['sentence_count']  = 2 * df['file_contents'].apply(num_sentences)
df['spelling_mistakes']  = -1 * df['file_contents'].apply(spelling_mistakes)
df['agreement']  = df['file_contents'].apply(agreement)
df['verbs']  = df['file_contents'].apply(verbs)
df['c3'] = 2 * df['c3']

print(df.columns)
df.to_csv('data.csv')