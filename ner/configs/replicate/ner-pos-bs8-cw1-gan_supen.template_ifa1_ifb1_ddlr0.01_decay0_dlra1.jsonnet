// jsonnet allows local variables like this
local cuda_device  = 0;
local train_size = -1;
local shuffle_id = 0;
local num_epochs = 140;
local patience = 18;
local grad_clipping = 5;
local grad_norm = 1;

local train_batch_size = 8;
local valid_batch_size = 256;

local constraints_wt = 1;
local implication_wt = 1;
local constraints_path = 'constraints.yml';

local embedding_dim = 256;
local hidden_dim = 128;
local num_lstm_layers = 1;
local lstm_dropout = 0.5;
local mlp1 = [[128, 0.3]];
local mlp2 = [[128, 0.3]];

local optimizer = "adam";
local learning_rate = 0.001;
local dd_optimizer = "sgd";
local dd_mom = 0.9;

local data_dir = '../data/ner-pos/gmb';
local train_data_file = 'train.txt.pkl';
local validation_data_file = 'dev.txt.pkl';
local unlabelled_train_data_file = 'train.txt.pkl';
local semi_supervised=false;
local which_mixer = 'cm';
local dd_increase_freq_after = 1;
local dd_increase_freq_by = 1;
local ddlr = 0.01;
local dd_decay_lr = 0; 
{
    "dd_decay_lr": dd_decay_lr,
    "dd_increase_freq_after": dd_increase_freq_after,
    "dd_increase_freq_by": dd_increase_freq_by, 
    "constraints_wt": constraints_wt,
    "dd_warmup_iters": 100,
    "dd_update_freq": 10,
    "which_mixer": which_mixer,
    "semi_supervised": semi_supervised,
    "unlabelled_train_data_file": unlabelled_train_data_file,
    "unlabelled_train_data_path": data_dir+'/shuffle'+shuffle_id+'/'+train_data_file,
    
    "serialization_dir": '../logs/replicate/ner-pos-gan-bs_8-sumpen-semi_'+semi_supervised+'-wm_'+which_mixer+'-ddoptim_'+dd_optimizer+'-ddlr_'+ddlr +'-ifa_'+dd_increase_freq_after+ '-ifb_'+ dd_increase_freq_by + '-decay_'+ dd_decay_lr + '-cw_'+constraints_wt+'/shuffle'+shuffle_id+'/ts'+train_size,
    "cuda_device": cuda_device,
    "data_dir": '../data/ner-pos/gmb',
    "train_data_file":  train_data_file,
    "validation_data_file": validation_data_file,
    "test_data_file": 'test.txt.pkl',
    "train_data_path": data_dir+'/shuffle'+shuffle_id+'/'+train_data_file,
    "validation_data_path": data_dir + '/shuffle'+shuffle_id+'/'+validation_data_file,
    "datasets_for_vocab_creation": ['train','unlabelled'],
    "dataset_reader": {
        "type": "mtl_reader",
        "how_many_sentences": train_size,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": 'token_ids'
            }
        }
    },
    "validation_dataset_reader": {
        "type": "mtl_reader",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": 'token_ids'
            }
        }
    },
    "unlabelled_dataset_reader": {
        "type": "mtl_reader",
        "start_from": train_size,
        "return_labels": false,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
                "namespace": 'token_ids'
            }
        }
    },

    "vocabulary": {
        "tokens_to_add": {
            "task1_labels": ['O', 'B-geo', 'B-org', 'B-tim', 'I-org', 'I-per', 'B-per', 'B-gpe', 'I-geo', 'I-tim', 'B-art', 'I-art', 'B-eve', 'I-eve', 'I-gpe', 'B-nat', 'I-nat'], 
            "task2_labels": ['NN', 'NNP', 'IN', 'DT', 'JJ', 'NNS', '.', 'VBD', ',', 'VBN', 'VBZ', 'VB', 'CD', 'CC', 'TO', 'RB', 'VBG', 'VBP', 'PRP', 'POS', 'PRP$', 'MD', 'WDT', 'JJS', 'JJR', 'LQU', 'NNPS', 'WP', 'RP', 'WRB', '$', 'RBR', 'RQU', ':', 'EX', 'LRB', 'RRB', 'RBS', ';', 'PDT', 'WP$', 'UH', 'FW']
        }
    },
    "dd_constraints": {
        "type": "mtl_constraints",
        "config": {
            "dd_constant_lambda": 0,
            "mtl_implication": {
                "constraints_path": constraints_path,
                "weight": implication_wt
            }
        }
    },

    "dd_optimizer": {
            "type": dd_optimizer,
            "lr": ddlr,
            "momentum": dd_mom
    },
    

    "model": {
        "type": "mtl_tagger",
        "text_field_embedder": {
            "type": 'basic',
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim,
                    "vocab_namespace": 'token_ids'
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim,
            "num_layers": num_lstm_layers,
            "dropout": lstm_dropout,
            "batch_first": true,
            "bidirectional": true
        },
        "mlp1": mlp1,
        "mlp2": mlp2,
        "task1_ignore_classes": ['O'],
        "task1_wts_uniform": false,
        "task2_wts_uniform": false,
        "task1_metric": 'f1',              
        "task2_metric": 'acc'             
    },
        
    "iterator": {
        "type": "bucket",
        "batch_size": train_batch_size,
        "sorting_keys": [["sentence", "num_tokens"]],
        "padding_noise": 0.2
    },
    "validation_iterator": {
        "type": "bucket",
        "batch_size": valid_batch_size,
        "sorting_keys": [["sentence", "num_tokens"]],
        "padding_noise": 0
    },
        
    "trainer": {
        "optimizer": {
            "type": optimizer,
            "lr": learning_rate
        },
        "validation_metric": '-nm',
        "patience": patience,
        "num_epochs": num_epochs,
        "cuda_device": cuda_device,
        "learning_rate_scheduler": {
            "type": 'reduce_on_plateau',
            "mode": 'min',
            "factor": 0.1, 
            "patience": 7,
            "verbose": true, 
            "threshold": 0.02,
            "threshold_mode": 'rel', 
            "cooldown": 0,
            "min_lr": 0.00001, 
            "eps": 1e-08
        },
        "should_log_learning_rate": true,
        "num_serialized_models_to_keep": 2,
        "grad_clipping": grad_clipping,
        "grad_norm": grad_norm,
        "summary_interval": 100
    }
}

