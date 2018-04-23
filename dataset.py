import unicodedata
import re

confusing_words=['and','your']
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def create_dataset(filename):
    with open(filename,'r',encoding='utf8') as f:
        input=f.readlines()
        with open("sample_input.csv",'w',encoding='utf8') as f1:
            f1.writelines(input)
create_dataset('training/news-commentary-v6.fr-en.en')

'''
with open('sample_output1.csv','r',encoding='utf8') as f:
    s=f.readLines()
with open('sample_output1.csv', 'w') as f:
    for line in s:
        s=unicodeToAscii(s)
        s=normalizeString(s)
        f.write(s)
'''
with open('sample_input.csv','r',encoding='utf8') as f:
    s=f.readlines()
with open('sample_input.csv', 'w') as f:
    for line in s:
        line1=unicodeToAscii(line)
        line1=normalizeString(line1)
        line1+="\n"
        try:
            lines1=line1.split('.')
            f.writelines(lines1)
        except Exception as e:
            pass


#confusing_words={'adapt':'adopt','adopt':'adapt','your':'you\'re','some':'sum','sum':'some','used':'sued','sued':'used','wired':'weird','weird':'wired'}
confusing_words={'soul':'sole','sole':'soul'}

with open('sample_output_soul.csv','w') as f2:
    with open('sample_input_soul.csv','w') as f1:
        with open("sample_input.csv",'r') as f:
            for line in f.readlines():
                for word in confusing_words:
                    if " "+word+" " in line:
                        f2.write(line)
                        line=line.replace(word,confusing_words[word])
                        f1.write(line)
                        break
