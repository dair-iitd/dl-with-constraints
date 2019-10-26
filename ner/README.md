# Named Entity Recognition (NER)

The code has been built on top of allennlp. 
Most of the files have been copied and modified from allennlp source code.

## Data
The data has been taken from [here](https://gmb.let.rug.nl/data.php), preprocessed and archived. 

Download [this]() tar file containing all the data and untar in the same directory as code. This will create a folder named `data` parallel to the `code` folder. It contains the gmb data preprocessed for our code. It already contains all the 10 shuffles which were used to run the experiments.

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
python dual_constrained_inference_find_optima.py -h #help
#For running on validation data
python dual_constrained_inference_find_optima.py --test 0 --ddlr 0.05 --dditer 25 --exp_dirs <exp_dir>
#For running on Test data:
python dual_constrained_inference_find_optima.py --test 1 --ddlr 0.05 --dditer 25 --exp_dirs <exp_dir>

#<exp_dir> : base serialization directory for a given configuration of hyper parameters as specified in the params_file while training the model. 
It should contain models for all the shuffles and all the training sizes for a given configuration. e.g.: ../logs/replicate/ner-pos-gan-bs_8-sumpen-semi_false-wm_cm-ddoptim_sgd-ddlr_0.01-ifa_1-ifb_1-decay_0-cw_1
```
It creates a pickle file named: `inference_lr0.05_iter25_test0.pkl` for run on validation data and `inference_lr0.05_iter25_test1.pkl` for test data in the `<exp_dir>`




