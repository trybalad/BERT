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
        self.nlp = spacy.load('pl_core_news_lg')
        self.vocab_size = 4
        self.word2index = {UNKNOWN_TOKEN: UNKNOWN_ID, CLASS_TOKEN: CLASS_ID,
                           SENTENCE_SEPARATOR_TOKEN: SENTENCE_SEPARATOR_ID, MASK_TOKEN: MASK_ID}
        self.index2word = {UNKNOWN_ID: UNKNOWN_TOKEN, CLASS_ID: CLASS_TOKEN,
                           SENTENCE_SEPARATOR_ID: SENTENCE_SEPARATOR_TOKEN, MASK_ID: MASK_TOKEN}

    def prepare_vocab(self, document_name, vocab_name, count_file):
        document = codecs.open(document_name, 'r', 'utf-8')
        vocab_document = codecs.open(vocab_name, 'w', 'utf-8')
        count_document = codecs.open(count_file, 'w', 'utf-8')

        count_words = {}
        line = document.readline()
        count = 0

        while line:
            line = line.strip()
            count += 1
            if line != '':
                for sentence in self.split_sentences(line):
                    self.learn_tokens(sentence.text, count_words, vocab_document)

            if count % 100 == 0:
                print(count, " lines read. Vocab size:", self.vocab_size)
            line = document.readline()
        for key, value in count_words.items():
            count_document.write(str(value))
            count_document.write(" ")
            count_document.write(key)
            count_document.write("\n")

    def change_to_reversible(self):
        if len(self.word2index) != len(self.index2word):
            self.index2word = {}
            for key, value in self.word2index.items():
                self.index2word[value] = key

    def read_vocab(self, vocab_name, reversible=False):
        document = codecs.open(vocab_name, 'r', 'utf-8')
        word = document.readline()
        while word:
            self.add_to_vocab(word, reversible)
            word = document.readline()

    def add_to_vocab(self, word, reversible=True):
        word = word.strip()
        if word not in self.word2index:
            self.word2index[word] = self.vocab_size
            if reversible:
                self.index2word[self.vocab_size] = word
            self.vocab_size += 1

    def write_vocab(self, vocab_name):
        document = codecs.open(vocab_name, 'w', 'utf-8')
        for key in self.word2index.keys():
            document.write(key + '\n')

    def split_sentences(self, document):
        doc = self.nlp(document)
        return doc.sents

    def convert_to_tokens(self, document, learn_vocab=False):
        doc = self.nlp(document)
        if learn_vocab:
            for token in doc:
                self.add_to_vocab(token.lemma_.lower(), False)
        return [token.lemma_.lower() for token in doc if not token.lemma_.isspace()]

    def convert_tokens_to_ids(self, tokens):
        result = []
        for token in tokens:
            result.append(self.word2index.get(token, UNKNOWN_ID))
        return result

    def convert_to_token(self, id):
        if len(self.index2word) != self.vocab_size:
            print("Make vocab reversible first")
            return ""
        return self.index2word[id]

    def learn_tokens(self, text, count_words, vocab_document):
        doc = self.nlp(text)
        for token in doc:
            token = token.lemma_.lower().strip()
            if token not in self.word2index:
                self.word2index[token] = self.vocab_size
                count_words[token] = 1
                self.vocab_size += 1
                vocab_document.write(token + '\n')
            else:
                count_words[token] = count_words[token] + 1
