import sys,os
sys.path.append(os.getcwd())

from keras_bert.data_generator import DataGenerator
from keras_bert.model import create_model
from keras_bert.tokenizer import Tokenizer
from keras_bert.training import train_model

"""Read vocab"""
print("Reading vocab.")
tokenizer = Tokenizer()
tokenizer.read_vocab('./data/vocab_wiki_small.txt')
tokenizer.change_to_reversible()
print("Vocab of size:", tokenizer.vocab_size, "loaded.")

"""Prepare data generators"""
data_generator = DataGenerator("./data/wiki_clean.txt", 32, tokenizer, batch_size=64, create_nsr_output=True)
print("Data generator prepared.")

"""Prepare model."""
sequence_encoder = create_model(tokenizer.vocab_size, 32, 768, 12, 12, 768)
print("Model created.")

"""Start training."""
train = train_model(sequence_encoder, 32, tokenizer, data_generator, epochs=500, checkpoint_file_path="$HOME/data/checkpoint_wiki_nsr.ckpt", load_checkpoint=False)

