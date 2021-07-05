from keras_bert.tokenizer import Tokenizer

tokenizer = Tokenizer()
print("Preparing vocab.")
tokenizer.prepare_vocab("./data/corpus_clean.txt", './data/counted_vocab.txt')
print("Vocab of size:", tokenizer.vocab_size, "created.")
