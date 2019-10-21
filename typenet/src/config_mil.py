import os

class Config_MIL:
    ''' Neural net hyperparameters '''

    gpu=True
    mention_rep = True


    ''' data hyperparameters '''



    def __init__(self, run_dir, args):

        self.dropout = args.dropout
        self.dataset = args.dataset
        self.args = args
        self.log_dir = args.log_dir 
        self.bag_size = args.bag_size

        self.lr = args.lr
        self.encoder = args.encoder
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.epsilon = args.epsilon
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.kernel_width = args.kernel_width
        self.clip_val = args.clip_val
        self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.save_model = args.save_model
        self.struct_weight = args.struct_weight
        self.linker_weight = args.linker_weight
        self.typing_weight = args.typing_weight


        self.complex = args.complex
        self.mode = args.mode
        self.bilinear_l2 = args.bilinear_l2
        self.parent_sample_size = args.parent_sample_size
        self.asymmetric = args.asymmetric

        self.test_batch_size = args.test_batch_size
        self.take_frac = args.take_frac
        self.use_transitive = args.use_transitive
        self.base_dir = args.base_dir
        base_dir = args.base_dir
        self.features = args.features

        self.feature_file = "%s/wiki_typing/feature2id_figer.txt" %base_dir
        
        self.override_lr  = args.override_lr 
        self.start_from_best = args.start_from_best
        self.ddlr = args.ddlr
        self.dd = args.dd
        self.inference_thr = args.inference_thr 
        self.dd_implication_wt = args.dd_implication_wt  
        self.dd_mutl_excl_wt  = args.dd_mutl_excl_wt 
        self.dd_mutl_excl_topk = args.dd_mutl_excl_topk 
        self.dd_update_freq = args.dd_update_freq
        self.dd_warmup_iter = args.dd_warmup_iter
        self.dd_tcc = args.dd_tcc 
        self.dd_weight_decay= args.dd_weight_decay  
        self.dd_optim = args.dd_optim 
        self.dd_mom = args.dd_mom  
        self.dd_logsig= args.dd_logsig  
        self.dd_penalty= args.dd_penalty 
        self.eval_topk = 0
        self.dd_constraint_wt = args.dd_constraint_wt
        self.dd_maxp_wt  =  args.dd_maxp_wt
        self.dd_maxp_thr =  args.dd_maxp_thr
        self.dd_constant_lambda =  args.dd_constant_lambda 
        
        self.dd_increase_freq_by = args.dd_increase_freq_by 
        self.dd_increase_freq_after = args.dd_increase_freq_after
        self.dd_decay_lr = args.dd_decay_lr 
        self.dd_decay_lr_after = args.dd_decay_lr_after 

        
        self.use_wt_as_lr_factor = args.use_wt_as_lr_factor 
        self.semi_mixer = args.semi_mixer
        self.semi_warmup_iter = args.semi_warmup_iter
        self.semi_sup = args.semi_sup
        self.class_imb = args.class_imb
        self.unlabelled_ratio = args.unlabelled_ratio 
    
        self.grad_norm = args.grad_norm
        self.grad_norm_before_warmup = args.grad_norm_before_warmup


# no support for struct weight with figer
        if self.dataset == "figer":
            self.test_bag_size = 1
            self.type_dict        = "%s//MIL_data/figer_type_dict.joblib" %base_dir
            self.typenet_matrix   = "%s/types_annotated/TypeNet_transitive_closure.joblib" %base_dir
            self.typenet_adj_matrix   = "%s/types_annotated/TypeNet_adj_matrix.joblib" %base_dir
            self.dev_file="%s/wiki_typing/dev"%base_dir
            self.test_file="%s/wiki_typing/test"%base_dir

        else:
            self.test_bag_size = 20
            self.type_dict        = "%s/MIL_data/TypeNet_type2idx.joblib" % base_dir
            self.typenet_matrix   = "%s/MIL_data/TypeNet_transitive_closure.joblib" % base_dir
            self.typenet_adj_matrix   = "%s/MIL_data/TypeNet_adj_matrix.joblib" % base_dir

            self.train_file = "%s/MIL_data/train.entities"%base_dir
            self.dev_file   = "%s/MIL_data/dev.entities" % base_dir
            self.test_file  = "%s/MIL_data/test.entities" % base_dir
            self.bag_file   = "%s/MIL_data/entity_bags.joblib" % base_dir
            self.entity_dict = "%s/MIL_data/entity_dict.joblib" % base_dir
            self.cross_wikis_shelve = "%s/MIL_data/alias_table.joblib" %base_dir



        self.embedding_file = "%s/data/pretrained_embeddings.npz" %base_dir
        self.embedding_downloaded_file = "%s/other_resources/glove.840B.300d.txt" %base_dir
        self.crosswikis_file = "%s/other_resources/dictionary.bz2"
        self.redirects_file = "/iesl/canvas/smurty/wiki-data/enwiki-20160920-redirect.tsv"


        self.entity_bags_dict = "%s/MIL_data/entity_bags.joblib" %base_dir

        if self.use_transitive:
            self.entity_type_dict = "%s/MIL_data/entity_%s_type_dict.joblib" %(base_dir, self.dataset)
        else:
            self.entity_type_dict = "%s/MIL_data/entity_type_dict_orig.joblib" %(base_dir)

        self.entity_type_dict_test = "%s/MIL_data/entity_%s_type_dict.joblib" %(base_dir, self.dataset)


        self.raw_entity_file="%s/AIDA_linking/AIDA_original/all_entities.txt"%base_dir
        self.raw_type_file="%s/types_annotated/typenet_structure.txt"%base_dir
        self.raw_entity_type_file="%s/AIDA_linking/AIDA_original/all_entities_types.txt"%base_dir


        self.vocab_file="%s/data/vocab.joblib"%base_dir
        #self.checkpoint_file = "%s/checkpoints" %run_dir
        self.model_name = args.model_name 
        self.checkpoint_file = "%s/%s" %(self.log_dir, args.model_name)
        self.model_file = 'best_params.pth'
        self.test_out_file = os.path.join(self.checkpoint_file, 'test_results.txt')
        try:
            os.makedirs(self.checkpoint_file)
        except:
            pass


