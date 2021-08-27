import os
import sys

sys.path.append(os.getcwd())
from keras_bert.prepare_data import create_tokens, create_masks, create_ids, create_segments, translate_ids, \
    create_pretrain_data
from keras_bert.data_generator import DataGenerator
from keras_bert.model import create_model
from keras_bert.tokenizer import Tokenizer
from keras_bert.training import train_model, prepare_pretrain_model_from_checkpoint
import numpy as np


print("Reading vocab.")
tokenizer = Tokenizer()
tokenizer.read_vocab('./data/counted_vocab.txt')
tokenizer.change_to_reversible()
print("Vocab of size:", tokenizer.vocab_size, "loaded.")

data_generator = DataGenerator("./data/validation_clean.txt", 32, tokenizer, batch_size=1, create_nsr_output=True)
print("Data generator prepared.")

sequence_encoder = create_model(tokenizer.vocab_size, 32, 512, 4, 4, 512)
print("Model created.")

# Start training.
model = prepare_pretrain_model_from_checkpoint(sequence_encoder, tokenizer, checkpoint_file_path="./data/checkpoint_test6.ckpt", load_checkpoint=True, old_checkpoint="./data/checkpoint_test5.ckpt")
model.evaluate(x=data_generator)

message = "Zazwyczaj dystrybucje są instalowane bezpośrednio na dysku twardym komputera. Istnieją jednak również dystrybucje, które da się uruchomić bezpośrednio z nośnika instalacyjnego."
tokens = create_tokens([message], tokenizer, 32)
mlm_tokens = create_pretrain_data(tokens, tokenizer)
ids = create_ids(mlm_tokens, 32, tokenizer)
mask = create_masks(mlm_tokens, 32)
segments = create_segments(mlm_tokens, 32)

result = model.predict(x = [np.array(ids), np.array(segments), np.array(mask)])
prediction = translate_ids(result, tokenizer)
print(tokens)
print(mlm_tokens)
print(prediction)

good = 0
all = 0
for i in range(0, len(tokens)):
    if tokens[i] != mlm_tokens[i]:
        all += 1
        if tokens[i] == prediction[i]:
            good += 1
print(good/all)