# Semantic Role Labeling (SRL)

The code has been built on top of allennlp. 
Most of the files have been copied and modified from allennlp source code.

## Data
We use English Ontonotes 5.0 dataset1 using the CONLL 2011/12 236 shared task format (Pradhan et al. [2012]) as the training data.
Since the data is not available publicly, we can not share it here.

## Environment
All experiments were run using `Python 3.6`. Use `requirements.txt` file to install all the required modules:
```
pip install -r requirements.txt
```

## Training the model
Code has been adapted from `allennlp`.
 Models can be trained using `dual_training.py` script, which primarily uses `gan_trainer.py` which is a modification of `Trainer` class in `allennlp`. For example, 
 ```
 python dual_training.py -h #help
 python dual_training.py --params_file ../configs/replicate_results/constr_all.template_pct10_ne200_gn3.5_ddlr0.05_ifb10_dlra10.jsonnet 
 ```
 
## Generating Params File with different configurations for the grid search
It is tedious and error-prone to create a `params_file` by hand for every different configuration. Hence, we have written a script, `create_configs.py`, to generate all the `params_file` used for the grid search over hyper-parameters. Certain variables are hard-coded in it. You may have to change them to generate a file for different configuration. 

```
python create_configs.py --template_params_file ../configs/replicate_results/constr_all.template --out_dir ../configs/replicate --exp_dir ../logs/replicate_results
```

## Collate Results: Viterbi Decoding, Violations and Evaluation using official perl script 
After all the models have been trained, use `eval_code/collate_with_metrics.py` script to collate all the results before and after viterbi decoding. It also computes the violation stats for each configuration of hyper parameters. For example:

```
python collate_with_metrics.py -h #help
#For running on validation data
python collate_with_metrics.py --archive_dir ../logs/replicate --is_parent 1 --out_file replicate_valid.csv --valid_data 1
#For running on Test data:
python collate_with_metrics.py --archive_dir ../logs/replicate --is_parent 1 --out_file replicate_test.csv --valid_data 0
# <archive_dir> : base directory containing all the serialization directories.
```

It runs three commands/scripts internally. 

`eval_custom.py`: iterate over test/valid data and run inference with and without viterbi decoding. Generates input for the perl script

`srl-eval.pl`: run the official evaluation script. 

`eval_custom.py`: iterate over test/valid data and run inference with and without viterbi decoding and compute constraint violations in the output.

Result is a csv file collating performance metrics and violation stats for all the serialization directories in the input base directory.
