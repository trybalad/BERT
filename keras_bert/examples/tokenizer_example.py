import spacy

nlp = spacy.load('pl_core_news_lg')
doc = nlp("Za oknem możemy zobaczyć piękne, jasnoniebieskie niebo.")
print([token for token in doc])
# [Za, oknem, możemy, zobaczyć, piękne, ,, jasnoniebieskie, niebo, .]

eng_sent = "Don't go there."
nlp = spacy.load("en_core_web_lg")
doc = nlp(eng_sent)
print([token for token in doc])
#[Do, n't, go, there, .]

print(eng_sent.split())
#["Don't", 'go', 'there.']