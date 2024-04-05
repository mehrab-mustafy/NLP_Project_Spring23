from sample_code_1 import num_sentences, spelling_mistakes
from sample_code_2 import agreement, verbs


def main():
    
    a = num_sentences()
    b = spelling_mistakes()
    c = agreement()
    d = verbs()
    this_input = input("Enter the Essay to grade: ")
    #sample input: this is my essay
    print(a, b, c, d, this_input)
    
    return 0



if __name__ == '__main__':
    exit(main())