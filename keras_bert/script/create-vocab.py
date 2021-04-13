from keras_bert.tokenizer import Tokenizer

tokenizer = Tokenizer()
print("Preparing vocab.")
tokenizer.prepare_vocab("./keras_bert/wiki_clean.txt", './data/vocab_wiki2.txt', './data/counted_vocab_wiki.txt')
print("Vocab of size:", tokenizer.vocab_size, "created.")
