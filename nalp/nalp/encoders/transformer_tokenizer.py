import tensorflow as tf
import tensorflow_text

import os
"""
    wiki_tokenizer is a BertTokenizer for transformer trained on tensorflow wikipedia dataset (wikipedia/20201201.en - size of 17.76 GiB)
    
    As a subword tokenizer, it has several tensorflow functions:
    tokenize(strings): tokenize the natural language to token_ids
    detokenize(tokenized): bring token_ids back to natural text
    lookup(token_ids): shows bert tokens that represents each token-ids
    get_vocab_size()
    get_vocab_path()
    get_reserved_tokens()
"""


class TransformerTokenizer:
    def __init__(self, tokenizers):
        self.tokenizers = tokenizers

    def tokenize(self, text):
        return self.tokenizers.en.tokenize(text).to_tensor()

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model_name = "wiki_tokenizers_2"
    tokenizers = tf.saved_model.load(model_name)

    tokenizer = TransformerTokenizer(tokenizers)
    print(tokenizer.tokenize(['hello tensorflow']))
    # print(tokenizers.en.tokenize(['hello tensorflow']))
    # print([item for item in dir(tokenizers.en) if not item.startswith('_')])

