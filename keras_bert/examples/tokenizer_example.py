import spacy

nlp = spacy.load('pl_core_news_lg')
doc = nlp("Za oknem możemy zobaczyć piękne, jasnoniebieskie niebo.")
print([token for token in doc])
# [Za, oknem, możemy, zobaczyć, piękne, ,, jasnoniebieskie, niebo, .]
