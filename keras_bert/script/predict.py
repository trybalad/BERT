from keras_bert.model import create_model
from keras_bert.prepare_data import create_tokens, create_masks, create_ids, create_segments, translate_ids, \
    create_pretrain_data
from keras_bert.tokenizer import Tokenizer
from keras_bert.training import prepare_pretrain_model_from_checkpoint

"""Read vocab"""
print("Reading vocab.")
tokenizer = Tokenizer()
tokenizer.read_vocab('./data/counted_vocab.txt')
print("Vocab of size:", tokenizer.vocab_size, "loaded.")

"""Prepare model."""
sequence_encoder = create_model(tokenizer.vocab_size, 16, 768, 16, 16, 2048)
print("Model created.")

"""Start predictions."""
model = prepare_pretrain_model_from_checkpoint(sequence_encoder, tokenizer,
                                               checkpoint_file_path="./data/checkpoint_d.ckpt", learn_type="mlp")

message = "Edytor tekstu asdf Office 2007 przewodnik dla gimnazjalisty Autor: Dariusz Kwieciński nauczyciel ZPO w Sieciechowie"
tokens = create_tokens([message], tokenizer, 16)
mask_t = create_pretrain_data(tokens, tokenizer)
ids = create_ids(mask_t, 16, tokenizer)
mask = create_masks(mask_t, 16)
segments = create_segments(mask_t, 16)

result = model.predict([ids, segments, mask])
print(tokens)
print(mask_t)
print(ids)
print(translate_ids(result, tokenizer))
