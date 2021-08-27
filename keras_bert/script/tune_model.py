from keras_bert.tuning.tuning_data_generator import TuningDataGenerator
from keras_bert.model import create_model
from keras_bert.tokenizer import Tokenizer
from keras_bert.tuning.fine_tuning import fine_tune

print("Reading vocab.")
tokenizer = Tokenizer()
tokenizer.read_vocab('./data/counted_vocab.txt')
tokenizer.change_to_reversible()
print("Vocab of size:", tokenizer.vocab_size, "loaded.")

data_generator = TuningDataGenerator("./data/ar/train.txt", 32, tokenizer, batch_size=64, tuning_type='ar')
print("Data generator prepared.")

sequence_encoder = create_model(tokenizer.vocab_size, 32, 512, 4, 4, 512)
print("Model created.")

# Start training.
train = fine_tune(sequence_encoder, 32, tokenizer, data_generator, epochs=10,
                    checkpoint_file_path="./data/ar/checkpoint_ar.ckpt", load_checkpoint=True, old_checkpoint="./data/checkpoint_test5.ckpt", learn_type='ar')
