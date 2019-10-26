# Named Entity Recognition (NER)

The code has been built on top of allennlp. 
Most of the files have been copied and modified from allennlp source code.

## Data
The data has been taken from [here](https://gmb.let.rug.nl/data.php), preprocessed and archived. 

Download [this](https://drive.google.com/open?id=1Qypy_JzocCqC1_-jPBf_5WKHqBVeY2lC) tar file containing all the data and untar in the same directory as code. This will create a folder named `data` parallel to the `code` folder. It contains the gmb data preprocessed for our code. It already contains all the 10 shuffles which were used to run the experiments.

```
tar -zxvf data.tar.gz
```

## Environment
All experiments were run using `Python 3.6`. Use `requirements.txt` file to install all the required modules:
```
pip install -r requirements.txt
```

## Training the model
Code has been adapted from `allennlp`.
 Models can be trained using `dual_training.py` script, which primarily uses `gan_trainer.py` which is a modification of `Trainer` class in `allennlp`. A list of training sizes and shuffle ids is provided as command line input. It enumerates over them and train a model for each configuration. For example, 
 ```
 python dual_training.py -h #help
 python dual_training.py --params_file ../configs/replicate/ner-pos-bs8-cw1-gan_supen_semi_min50p.template_ifa1_ifb1_ddlr0.05_decay0_dlra1.jsonnet --train_size_list 400 800 1600 --shuffle_id_list 1 2 3 4
 ```
 
## Generating Params File with different configurations for the grid search
It is tedious and error-prone to create a `params_file` by hand for every different configuration. Hence, we have written a script, `create_configs.py`, to generate all the `params_file` used for the grid search over hyper-parameters. Certain variables are hard-coded in it. You may have to change them to generate a file for different configuration. 

```
python create_configs.py --template_params_file ../configs/replicate/ner-pos-bs8-cw1-gan_supen.template --out_dir ../configs/replicate --exp_dir ../logs/replicate
```

## Constrained Inference 
After all the models have been trained, use `dual_decomposition_inference.py` script to run dual decomposition constrained inference procedure. For example:
```
python dual_decomposition_inference.py -h #help
#For running on validation data
python dual_decomposition_inference.py --test 0 --ddlr 0.05 --dditer 25 --exp_dirs <exp_dir>
#For running on Test data:
python dual_decomposition_inference.py --test 1 --ddlr 0.05 --dditer 25 --exp_dirs <exp_dir>

#<exp_dir> : base serialization directory for a given configuration of hyper parameters as specified in the params_file while training the model. 
It should contain models for all the shuffles and all the training sizes for a given configuration. e.g.: ../logs/replicate/ner-pos-gan-bs_8-sumpen-semi_false-wm_cm-ddoptim_sgd-ddlr_0.01-ifa_1-ifb_1-decay_0-cw_1
```
It creates a pickle file named: `inference_lr0.05_iter25_test0.pkl` for run on validation data and `inference_lr0.05_iter25_test1.pkl` for test data in the `<exp_dir>`

##Collate results
`dual_decomposition_inference.py collates all the results when `collate` flag is on and `exp_dirs` contains a list of all the directories (one per configuration of hyper-parameters)

We need to first run it on `dev` data so that best hyper-parameters for `test` can be extracted. List of configuration names  (identified by the directory names) amongst which the best is chosen for the test is currently hard-coded (`cl_list` and `slmin50p_list`).  You may have to modify them before running.

To run:
```
#For running on validation data
python dual_decomposition_inference.py --test 0 --ddlr 0.05 --dditer 25 --exp_dirs <space_separarted_list_of_exp_dir> --collate 1 --output_file ../results/replicate
#For running on Test data:
python dual_decomposition_inference.py --test 1 --ddlr 0.05 --dditer 25 --exp_dirs <space_seperated_list_of_exp_dir> --output_file ../results/replicate
```

It outputs a bunch of files containing all the performance metrics. 

`<output_file>_<ddlr>_<dditer>.csv`: Contains the best configuration for constrained learning and semi-supervision for all training sizes.

`<output_file><ddlr>_<dditer>_test_<test>.csv-plot-inference.csv`: Use to plot performance as a function of #iterations of dual decomposition.

`<output_file><ddlr>_<dditer>_test_<test>.csv-plot-summary_niterbest.csv`: Contains various metrics for each configuration before and after dual decomposition.





