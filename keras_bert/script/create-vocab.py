from keras_bert.tokenizer import Tokenizer

tokenizer = Tokenizer()
print("Preparing vocab.")
tokenizer.prepare_vocab("../../data/pl_dedup.txt")
tokenizer.write_vocab('../../data/vocab.txt')
print("Vocab of size:", tokenizer.vocab_size, "created.")
