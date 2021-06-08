"""Config
"""

# pylint: disable=invalid-name

import copy

# Total number of training epochs (including pre-train and full-train)
max_nepochs = 30
pretrain_nepochs = 10  # Number of pre-train epochs (training as autoencoder)
display = 500  # Display the training results every N training steps.
# Display the dev results every N training steps (set to a
# very large value to disable it).
display_eval = 1e10

sample_path = './samples'
checkpoint_path = './checkpoints'
restore = ''   # Model snapshot to restore from

lambda_g = .1  # Weight of the classification loss
gamma_decay = 0.5  # Gumbel-softmax temperature anneal rate

train_data = {
    'batch_size': 64,
    'source_dataset': {
        "files": 'wiki_vitranslated_sentences.txt',
        'vocab_file': 'vocab.txt'
    },
    'target_dataset': {
        "files": 'wiki_good_sentences.txt',
        'vocab_file': 'vocab.txt'
    }
}

val_data = copy.deepcopy(train_data)

test_data = copy.deepcopy(train_data)

model = {
    'dim_c': 200,
    'dim_z': 500,
    'embedder': {
        'dim': 100,
    },
    'encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700
            },
            'dropout': {
                'input_keep_prob': 0.5
            }
        }
    },
    'decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'BahdanauAttention',
            'kwargs': {
                'num_units': 700,
            },
            'attention_layer_size': 700,
        },
        'max_decoding_length_train': 150,
        'max_decoding_length_infer': 150,
    },
    'classifier': {
        'kernel_size': [3, 4, 5],
        'filters': 128,
        'other_conv_kwargs': {'padding': 'same'},
        'dropout_conv': [1],
        'dropout_rate': 0.5,
        'num_dense_layers': 0,
        'num_classes': 1
    },
    'opt': {
        'optimizer': {
            'type':  'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
    },
}
