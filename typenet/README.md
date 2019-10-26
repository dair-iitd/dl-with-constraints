# Fine Grained Entity Typing (Typenet)

The code has been adapted from the baseline code available at: https://github.com/MurtyShikhar/Hierarchical-Typing

## Data
The data has been taken from the baseline repo and has been rearranged so that all experiments can be replicated seamlessly.

Download [this](https://drive.google.com/open?id=1jntle0aCdXC-V7NKGxZ7Q9qExudWPuW9) tar file containing all the data and untar in the same directory as src. This will create a folder named `resources` parallel to the `src` folder.

```
tar -zxvf resources.tar.gz
```

## Environment
To clone the conda environment, run:

```
conda env create -f environment.yml
conda activate py2
```

## Training the model
 Models can be trained using `deploy_mil_constraints.py` script. For example, baseline model using 5% data can be trained by running:
 
 ```
 python deploy_mil_constraints.py -dataset typenet -base_dir ../resources -take_frac 0.0500 -linker_weight 0.0 -clip_val 10 -batch_size 10 -mode typing -embedding_dim 300 -lr 0.001 -beta1 0.9 -beta2 0.999 -epsilon 1e-4 -typing_weight 1 -test_batch_size 20 -save_model 1 -log_dir ../logs_0.0500 -dd -num_epochs 150 -weight_decay 1e-6 -asymmetric 0 -dd_update_freq 10 -dd_weight_decay 0.0 -dd_increase_freq_after 1 -dd_mutl_excl_wt 0 -semi_mixer cm -struct_weight 0 -ddlr 0 -semi_sup 0 -dd_penalty mix -grad_norm 0.15 -dropout 0.5 -use_transitive 0 -dd_optim sgd -complex 0 -use_wt_as_lr_factor 0 -encoder basic -unlabelled_ratio 20 -dd_logsig 0 -dd_warmup_iter 1000 -dd_implication_wt 0 -dd_increase_freq_by 0 -dd_constant_lambda 0 -dd_mom 0.0 -dd_maxp_wt 0 -model_name c-0_conlam-0_dimpwt-0_ifb-0_dls-0_dmaxpwt-0_dm-0.0_dmutexwt-0_do-sgd_dp-mix_warmup-1000_ddlr-0_drop-0.5_e-basic_gn-0.15_smix-cm_semi-0_sw-0_ulrat-20_ut-0_uwlr-0 
 ```
## Generating commands for the grid search
It is tedious to write the above command by hand everytime we need to train a model. Hence, we have written a script to generate all the commands to do the grid search over hyper-parameters. For example, the command below will generate a file named `5p_dair_0.sh` containing all the commands for training models using 5% data. All the logs will be generated in `../logs_0.0500` directory.
This has also been adapted from a similar script in the baseline repo.

```
python generate_commands_for_grid_search.py -log_dir ../logs -base_dir ../resources -dataset typenet -take_frac 0.05 -file_name 5p -save_model 1 -num_streams 1
```

## Collate results
Use `extract_scores_and_plot.py` to collate all the results. 

```
python extract_scores_and_plot.py -h # help
python extract_scores_and_plot.py -input_dir ../logs_0.0500 -output_file <output_file_name_prefix>
```

It outputs a bunch of files prefixed `<output_file>` to their names. 

`<output_file>_pivot.csv`:  contains summary

`<output_file>_plots.pdf`: plots performance metric and learning rate as a function of number of iterations. 

