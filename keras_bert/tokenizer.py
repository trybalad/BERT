import codecs

import spacy

UNKNOWN_TOKEN = "[UNK]"
UNKNOWN_ID = 0
CLASS_TOKEN = "[CLS]"
CLASS_ID = 1
SENTENCE_SEPARATOR_TOKEN = "[SEP]"
SENTENCE_SEPARATOR_ID = 2
MASK_TOKEN = "[MASK]"
MASK_ID = 3


class Tokenizer():
    def __init__(self):
        self.nlp = spacy.load('pl_core_news_md')
        self.vocab_size = 4
        self.word2index = {UNKNOWN_TOKEN: UNKNOWN_ID, CLASS_TOKEN: CLASS_ID,
                           SENTENCE_SEPARATOR_TOKEN: SENTENCE_SEPARATOR_ID, MASK_TOKEN: MASK_ID}
        self.index2word = {UNKNOWN_ID: UNKNOWN_TOKEN, CLASS_ID: CLASS_TOKEN,
                           SENTENCE_SEPARATOR_ID: SENTENCE_SEPARATOR_TOKEN, MASK_ID: MASK_TOKEN}

    def prepare_vocab(self, document_name):
        document = codecs.open(document_name, 'r', 'utf-8')
        lines = document.readlines()
        lines = [line.strip() for line in lines if line.strip() != '']
        for line in lines:
            self.convert_to_tokens(line)

    def split_sentences(self, document):
        doc = self.nlp(document)
        return doc.sents

    def convert_to_tokens(self, document):
        doc = self.nlp(document)
        for token in doc:
            if token.lemma_ not in self.word2index:
                self.word2index[token.lemma_] = self.vocab_size
                self.index2word[self.vocab_size] = token.lemma_
                self.vocab_size += 1
        return [token.lemma_ for token in doc]