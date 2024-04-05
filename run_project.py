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


def main():
    
    #this_input = input("Enter the Essay to grade: ")
    text = 'Finaly, people were being successful or not is depen on what is your personal atetivies. Somehow people sucessful just for their own good, but many of other successful people I think will probably pay back for the socity, so i still believed people suessful is not just for thier own good is for every one in the would.'
    a = num_sentences(text)
    b = spelling_mistakes(text)
    # c_1, c_2, c_3 = agreement()
    # d_1, d_2 = verbs()
    print(a, b)
    # Final_Score = 2*a - b + c_1 + c_2 + 2*c_3 + 3*d_1 + 2*d_2
    # print(f"The final score of this essay is: {Final_Score}")
    # print("This essay is classified as: high")
    # print("This essay is classified as: low")
    return 0



if __name__ == '__main__':
    exit(main())