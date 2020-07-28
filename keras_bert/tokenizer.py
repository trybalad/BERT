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


class Tokenizer:
    def __init__(self):
        self.nlp = spacy.load('pl_core_news_md')
        self.vocab_size = 4
        self.word2index = {UNKNOWN_TOKEN: UNKNOWN_ID, CLASS_TOKEN: CLASS_ID,
                           SENTENCE_SEPARATOR_TOKEN: SENTENCE_SEPARATOR_ID, MASK_TOKEN: MASK_ID}
        self.index2word = {UNKNOWN_ID: UNKNOWN_TOKEN, CLASS_ID: CLASS_TOKEN,
                           SENTENCE_SEPARATOR_ID: SENTENCE_SEPARATOR_TOKEN, MASK_ID: MASK_TOKEN}

    def prepare_vocab(self, document_name):
        document = codecs.open(document_name, 'r', 'utf-8')
        line = document.readline()

        while line:
            line = line.strip()
            if line != '':
                for sentence in self.split_sentences(line):
                    self.convert_to_tokens(sentence.text, True)
            line = document.readline()

    def read_vocab(self, vocab_name):
        document = codecs.open(vocab_name, 'r', 'utf-8')
        word = document.readline()
        while word:
            self.add_to_vocab(word)
            word = document.readline()

    def add_to_vocab(self, word):
        word = word.strip()
        if word not in self.word2index:
            self.word2index[word] = self.vocab_size
            self.index2word[self.vocab_size] = word
            self.vocab_size += 1

    def write_vocab(self, vocab_name):
        document = codecs.open(vocab_name, 'w', 'utf-8')
        document.write(self.index2word[0])
        for i in range(1, self.vocab_size):
            document.write('\n' + self.index2word[i])

    def split_sentences(self, document):
        doc = self.nlp(document)
        return doc.sents

    def convert_to_tokens(self, document, learn_vocab=False):
        doc = self.nlp(document)
        if learn_vocab:
            for token in doc:
                self.add_to_vocab(token.lemma_)
        return [token.lemma_ for token in doc if not token.lemma_.isspace()]

    def convert_tokens_to_ids(self, tokens):
        result = []
        for token in tokens:
            result.append(self.word2index[token])
        return result

    def convert_to_token(self, id):
        return self.index2word[id]
