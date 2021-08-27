import spacy

nlp = spacy.load("pl_core_news_lg")
doc = nlp("Firma Microsoft otwiera nowy oddział w Krakowie. "
          "Siedziba firmy będzie znajdować się na ulicy "
          "Krowoderskiej zaraz obok sklepu Żabka.")

for ent in doc.ents:
    print(ent.text, ent.label_)
# Microsoft orgName
# Krakowie placeName
# ulicy Krowoderskiej geogName
# Żabka orgName
