import re
from konlpy.tag import Okt


okt = Okt()
tokenized = []
for text in corpus:
    for line in text.split('.'):
        line = re.sub(r'[^가-힣]', '', line)
        line = okt.morphs(line)
        tokenized.append(line)
print(tokenized)
