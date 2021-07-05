import spacy

nlp = spacy.load("pl_core_news_lg")
doc = nlp("Brązowy pies bardzo głośno szczekał na dziwną osobe ubraną w szary płaszcz.")
print([token.lemma_ for token in doc])
print(doc.noun_chunks)
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
            chunk.root.head.text)