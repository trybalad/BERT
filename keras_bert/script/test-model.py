from keras_bert.data_generator import DataGenerator
from keras_bert.model import create_model
from keras_bert.tokenizer import Tokenizer
from keras_bert.training import train_model

"""Read vocab"""
print("Reading vocab.")
tokenizer = Tokenizer()
tokenizer.read_vocab('../../data/vocab.txt')
print("Vocab of size:", tokenizer.vocab_size, "loaded.")

"""Prepare data generators"""
data_generator = DataGenerator("../../data/pl_dedup.txt", 64, tokenizer, batch_size=64, create_nsr_output=False)
print("Data generator prepared.")

"""Prepare model."""
sequence_encoder = create_model(tokenizer.vocab_size, 64, 768, 16, 16, 2048)
print("Model created.")

"""Start training."""
train = train_model(sequence_encoder, 64, tokenizer, data_generator, epochs=4, checkpoint_file_path="../../data/checkpoint.ckpt", load_checkpoint=False, learn_type="mlp")

