# This is a tutorial on using this library
# first off we need a text_encoder so we would know our vocab_size (and later on use it to encode sentences)

from bert.tokenization import FullTokenizer
from keras_bert.model import create_model
from keras_bert.prepare_data import create_tokens
from keras_bert.training import train_model

# nlp = spacy.load("pl_model")
# doc = nlp("To jest jedno zdanie. A to jest inne.")
# for sent in doc.sents:
#     print(sent.text)

# sentence_piece_encoder = BERTTextEncoder(vocab_file='../data/bert_pl_model/vocab.txt')

# now we need a sequence encoder

tokenizer = FullTokenizer(vocab_file='../data/bert_pl_model/vocab.txt')
sequence_encoder = create_model(len(tokenizer.vocab), 128, 256, 12, 8, 2048)
print(sequence_encoder.summary())


tokens = create_tokens("../data/Document.txt")
tokens = tokens[:10]
train_model(sequence_encoder, tokens, 128, len(tokenizer.vocab))
# ids = create_ids(tokens, 512)
# segments = create_segments(tokens, 512)
# masks = create_masks(tokens, 512)
# for i in range(len(tokens)):
#     print(tokens[i])
#     print(ids[i])
#     print(segments[i])
#     print(masks[i])
