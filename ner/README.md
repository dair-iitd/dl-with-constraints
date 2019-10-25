# Named Entity Recognition (NER)

The code has been built on top of allennlp. 
Most of the files have been copied and modified from allennlp source code.

## Data
The data has been taken from [here](https://gmb.let.rug.nl/data.php). 

Download [this]() tar file containing all the data and untar in the same directory as code. This will create a folder named `data` parallel to the `code` folder. It contains the gmb data preprocessed for our code. It already contains all the 10 shuffles where were used to run the experiments.

```
wget ()

tar -zxvf data.tar.gz
```

## Environment
All experiments were run using `Python 3.6`. Use `requirements.txt` file to install all the required modules:
```
pip install -r requirements.txt
```

## Training the model
 Models can be trained using `dual_training.py` script. It enumerates over all the input training sizes and input shuffles and train a model for each configuration. For example, 
 ```
 python dual_training.py -h #help
 python deploy_mil_constraints.py -dataset typenet -base_dir ../resources -take_frac 0.0500 -linker_weight 0.0 -clip_val 10 -batch_size 10 -mode typing -embedding_dim 300 -lr 0.001 -beta1 0.9 -beta2 0.999 -epsilon 1e-4 -typing_weight 1 -test_batch_size 20 -save_model 1 -log_dir ../logs_0.0500 -dd -num_epochs 150 -weight_decay 1e-6 -asymmetric 0 -dd_update_freq 10 -dd_weight_decay 0.0 -dd_increase_freq_after 1 -dd_mutl_excl_wt 0 -semi_mixer cm -struct_weight 0 -ddlr 0 -semi_sup 0 -dd_penalty mix -grad_norm 0.15 -dropout 0.5 -use_transitive 0 -dd_optim sgd -complex 0 -use_wt_as_lr_factor 0 -encoder basic -unlabelled_ratio 20 -dd_logsig 0 -dd_warmup_iter 1000 -dd_implication_wt 0 -dd_increase_freq_by 0 -dd_constant_lambda 0 -dd_mom 0.0 -dd_maxp_wt 0 -model_name c-0_conlam-0_dimpwt-0_ifb-0_dls-0_dmaxpwt-0_dm-0.0_dmutexwt-0_do-sgd_dp-mix_warmup-1000_ddlr-0_drop-0.5_e-basic_gn-0.15_smix-cm_semi-0_sw-0_ulrat-20_ut-0_uwlr-0 
 ```
## Generating commands for the grid search
It is tedious to write the above command by hand everytime we need to train a model. Hence, we have written a script to generate all the commands to do the grid search over hyper-parameters. For example, the command below will generate a file named `5p_dair_0.sh` containing all the commands for training models using 5% data. All the logs will be generated in `../logs_0.0500` directory.
This has also been adapted from a similar script in the baseline repo.

```
python generate_commands_for_grid_search.py -log_dir ../logs -base_dir ../resources -dataset typenet -take_frac 0.05 -file_name 5p -save_model 1 -num_streams 1
```
