import pandas as pd
import os
from sample_code_1 import num_sentences, spelling_mistakes
from sample_code_2 import agreement, verbs
from syntactic_wellformedness import get_wellformedness
from embeddings_d import cosine_similarity_prompt_essay, get_d2

def read_file_contents(filename, directory):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        return file.read()
    
    
df = pd.read_csv('index.csv', delimiter=';')
directory = 'essays/'        
df['file_contents'] = df.apply(lambda row: read_file_contents(row['filename'], directory), axis=1)
df['file_contents'] = df['file_contents'].str.replace('\n', '').str.replace('\t', '').str.replace("'", '')
df['file_contents'] = df['file_contents'].str.replace(r'\s+', ' ', regex=True)
df['sentence_count']  = 2 * df['file_contents'].apply(num_sentences)
df['spelling_mistakes']  = -1 * df['file_contents'].apply(spelling_mistakes)
df['agreement']  = df['file_contents'].apply(agreement)
df['verbs']  = df['file_contents'].apply(verbs)
df['c3'] = 2 * df['file_contents'].apply(get_wellformedness)
for i in range(len(df)):
    prompt = df.at[i, 'prompt']
    essay = df.at[i, 'file_contents']
    df.at[i,'d1'] = 3 * cosine_similarity_prompt_essay(prompt, essay)
    
    
df['d2'] = 1 * df['file_contents'].apply(get_d2)

print(df.columns)
df.to_csv('data.csv')