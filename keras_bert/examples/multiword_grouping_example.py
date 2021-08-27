import spacy


def group_words(text):
    list_string = text.split(' ')
    group = '-'.join(list_string)

    return group


nlp = spacy.load("pl_core_news_lg")
text = "Jaś wysłał kartkę z Nowego Jorku do swojego wujka " \
       "mieszkającego w Starym Sączu."
doc = nlp(text)

for ent in reversed(doc.ents):
    text = text[:ent.start_char] \
           + group_words(ent.text) \
           + text[ent.end_char:]
print(text)
# Jaś wysłał kartkę z Nowego-Jorku do swojego wujka
# mieszkającego w Starym-Sączu.
