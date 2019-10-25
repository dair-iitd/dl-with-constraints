// jsonnet allows local variables like this

local percent_data = 5;

local dd_increase_freq_by  = 1;
local dd_decay_lr_after = 10;
local grad_norm = 3.5;
local num_epochs = 200;
local ddlr = 0.05;


local which_mixer = 'cm';
local cuda_device  = 0;
local unlabelled_percent_data = 100 - percent_data;

local label_encoding = "BIOUL";
local lazy = false;

local semi_supervised = false;

local trans_aggregate_type = "sum";
local span_aggregate_type = "sum";

local span_wt = 1;
local terminal_wt = 1;

local dd_optimizer = "sgd";
local dd_mom = 0.9;

local dd_warmup_iters = 432;
local dd_increase_freq_after  = 1;
local constraints_wt = 1;


{
	"dd_decay_lr": 2,
    "dd_increase_freq_after": dd_increase_freq_after,
	"dd_increase_freq_by": dd_increase_freq_by,
    "dd_decay_lr_after": dd_decay_lr_after,
    "which_mixer": which_mixer,
    "semi_supervised": semi_supervised,
    "cuda_device": cuda_device,
    "dd_warmup_iters": dd_warmup_iters,
    "dd_update_freq": 10,
    "constraints_wt": constraints_wt,   
    "calc_valid_freq": 1, 
    "grad_norm_before_warmup": 1,

	"serialization_dir": "../logs/replicate/constr_glove_ifb"+dd_increase_freq_by+"_pct"+percent_data+"_gn"+grad_norm+"_dlra"+dd_decay_lr_after+"_ddlr"+ddlr+"_ne"+num_epochs,

	"datasets_for_vocab_creation": ['train','unlabelled'],

	"dataset_reader":
	{
		"type":"srl_custom",
		"label_encoding": label_encoding,
		"percent_data": percent_data,
	  	"token_indexers": {
			"tokens": {
			   "type": "single_id",
			   "lowercase_tokens": true
			},
			"elmo": {
				"type": "elmo_characters"
			}
	    },

	    "lazy": lazy
	},

	"unlabelled_dataset_reader": 
	{
		"type":"srl_custom",
		"label_encoding": label_encoding,
        "return_labels": false,
		"percent_data": unlabelled_percent_data,
	  	"token_indexers": {
			"tokens": {
			   "type": "single_id",
			   "lowercase_tokens": true
			},
			"elmo": {
				"type": "elmo_characters"
			}
	    },

		"lazy": lazy
	}, 

	"train_data_path": "../data/dataset/ONTONOTES_CONLL/conll-formatted-ontonotes-5.0/data/train",
	"unlabelled_train_data_path": "../data/dataset/ONTONOTES_CONLL/conll-formatted-ontonotes-5.0/data/train",
	"validation_data_path": "../data/dataset/ONTONOTES_CONLL/conll-formatted-ontonotes-5.0/data/development/",
	"test_data_path": "../data/dataset/ONTONOTES_CONLL/conll-formatted-ontonotes-5.0/data/test/", 
	
	"model": 
	{
		"type": "srl_custom", 
		"label_encoding": label_encoding,
		"text_field_embedder": 
		{
			"tokens": {
		        "type": "embedding",
		        "embedding_dim": 100,
		        "pretrained_file": "../data/glove/glove.6B.100d.txt",
		        "trainable": true
		      },

			"elmo": 
			{
				"type": "elmo_token_embedder", 
				"options_file": "../data/elmo_weights/fta/elmo_2x4096_512_2048cnn_2xhighway_options.json", 
				"weight_file": "../data/elmo_weights/fta/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5", 
				"do_layer_norm": false, 
				"dropout": 0.5
			}
		}, 

		"initializer": [["tag_projection_layer.*weight", {"type": "orthogonal"}]], 
		"encoder": 
		{
			"type": "alternating_lstm", 
			"input_size": 1224, 
			"hidden_size": 300, 
			"num_layers": 8, 
			"recurrent_dropout_probability": 0.1, 
			"use_highway": true			
		}, 

		"binary_feature_dim": 100, 		
	}, 

	"dd_constraints": {
        "type": "srl_constraints_cagg_choice",
        "config": {
            "dd_constant_lambda": 0,
            "span_implication_bl": {                
                "weight": span_wt,
                "aggregate_type": span_aggregate_type,
                "use_maxp": false
            },
            "span_implication_lb": {                
                "weight": span_wt,
                "aggregate_type": span_aggregate_type,
                "use_maxp": false
            },
            "terminal_tag_b": {                
                "weight": terminal_wt
            },
            "terminal_tag_l": {                
                "weight": terminal_wt
            },
            "transition": {
            	"weight": 1,
            	"aggregate_type": trans_aggregate_type,
                "include_eos": 0
            }
        }
    }, 

    "dd_optimizer": {
            "type": dd_optimizer,
            "lr": ddlr,
            "momentum": dd_mom,
    },

	"iterator": 
	{
		"type": "bucket", 
		"sorting_keys": [["tokens", "num_tokens"]], 
		"batch_size": 40
	}, 

	"validation_iterator": 
	{
		"type": "bucket", 
		"sorting_keys": [["tokens", "num_tokens"]], 
		"batch_size": 80
	}, 

	"trainer": 
	{
		"num_epochs": num_epochs, 
		"grad_norm": grad_norm,
		"grad_clipping": 1.0, 
		"patience": 200, 
		"num_serialized_models_to_keep": 2, 
		"validation_metric": "+f1-measure-overall", 
		"cuda_device": 0, 
		"should_log_learning_rate": true,			
		"optimizer": 
		{
			"type": "adadelta", 
			"rho": 0.95,
		}
	}, 

	"vocabulary": 
	{
		"directory_path": "../data/elmo_weights/vocabulary/"
	}
}


