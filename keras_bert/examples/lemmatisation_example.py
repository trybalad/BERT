import spacy

nlp = spacy.load('pl_core_news_lg')
doc = nlp("Za oknem możemy zobaczyć piękne, jasnoniebieskie niebo.")
print([token.lemma_ for token in doc])
# ['za', 'okno', 'móc', 'zobaczyć', 'piękny', ',', 'jasnoniebieski', 'niebo', '.']
