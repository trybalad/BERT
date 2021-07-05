import spacy

nlp = spacy.load('pl_core_news_lg')
stop_words = nlp.Defaults.stop_words

doc = nlp("Pies szczekał, ponieważ zobaczył kogoś obcego.")
print(len(stop_words))  # 381
print([token for token in doc if token.lemma_ not in stop_words])
# [Pies, szczekał, ,, zobaczył, obcego, .]
