from math import log10


def calculate_tf(term, doc):
    size = len(doc)  # D
    count = 0  # T
    for token in doc:
        if token == term:
            count += 1
    return count / size  # TF = T/D


def calculate_idf(docs, dict):
    idf = {}
    counts = {}  # Nt
    size = len(docs)  # N

    for term in dict:
        counts[term] = 0
        for doc in docs:
            for token in doc:
                if token == term:
                    counts[term] += 1
                    break

    for term in dict:
        idf[term] = log10(size / counts[term])
    return idf


def calculate_tf_idf(term, doc, idf):
    return idf[term] * calculate_tf(term, doc)


dict = ["celsjusz", "czarna", "herbata", "parzyć", "stopień",
        "temperatura", "w", "zielona", "70", "100"]
doc1 = ["herbata", "zielona", "parzyć", "w", "temperatura",
        "70", "stopień", "celsjusz"]
# Herbatę zieloną parzymy w temperaturze 70 stopni Celsjusza.
doc2 = ["herbata", "czarna", "parzyć", "w", "temperatura",
        "100", "stopień", "celsjusz"]
# Herbatę czarną parzymy w temperaturze 100 stopni Celsjusza.
dataset = [doc1, doc2]

idf = calculate_idf(dataset, dict)

tfidf = calculate_tf_idf("czarna", doc1, idf)
print(tfidf)  # 0.0

tfidf = calculate_tf_idf("czarna", doc2, idf)
print(tfidf)  # 0.03762874945799765

tfidf = calculate_tf_idf("herbata", doc1, idf)
print(tfidf)  # 0.0

tfidf = calculate_tf_idf("herbata", doc2, idf)
print(tfidf)  # 0.0