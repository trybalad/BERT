import os
import sys

sys.path.append(os.getcwd())

from keras_bert.data_generator import DataGenerator
from keras_bert.model import create_model
from keras_bert.tokenizer import Tokenizer
from keras_bert.training import train_model

print("Reading vocab.")
tokenizer = Tokenizer()
tokenizer.read_vocab('./data/counted_vocab.txt')
tokenizer.change_to_reversible()
print("Vocab of size:", tokenizer.vocab_size, "loaded.")

data_generator = DataGenerator("./data/corpus_clean.txt", 32, tokenizer, batch_size=16, create_nsr_output=True)
print("Data generator prepared.")

sequence_encoder = create_model(tokenizer.vocab_size, 32, 512, 4, 4, 512)
print("Model created.")

# Start training.
train = train_model(sequence_encoder, 32, tokenizer, data_generator, epochs=500,
                    checkpoint_file_path="./data/checkpoint_test.ckpt", load_checkpoint=False)
