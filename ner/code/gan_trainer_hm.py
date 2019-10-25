"""
Code is copied from allennlp.Trainer and modified to suit our need for training with constraints 
"""

from IPython.core.debugger import Pdb
import logging
import math
import os
import time
import re
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, NamedTuple
from collections import defaultdict

import torch
import torch.optim.lr_scheduler
import shutil 
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device 
from allennlp.common.util import (dump_metrics, gpu_memory_mb, peak_memory_mb,get_frozen_and_tunable_parameter_names, lazy_groups_of)
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer_base import TrainerBase
from allennlp.training import util as training_util
from allennlp.training.moving_average import MovingAverage
from allennlp.data.dataset_readers import DatasetReader
from allennlp.training.trainer import Trainer 
import numpy
import tracemalloc

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class TrainerPiecesForSemi(NamedTuple):
    """
    We would like to avoid having complex instantiation logic taking place
    in `Trainer.from_params`. This helper class has a `from_params` that
    instantiates a model, loads train (and possibly validation and test) datasets,
    constructs a Vocabulary, creates data iterators, and handles a little bit
    of bookkeeping. If you're creating your own alternative training regime
    you might be able to use this.
    """
    model: Model
    iterator: DataIterator
    train_dataset: Iterable[Instance]
    validation_dataset: Iterable[Instance]
    test_dataset: Iterable[Instance]
    validation_iterator: DataIterator
    unlabelled_dataset: Iterable[Instance]
    params: Params

    @staticmethod
    def from_params(params: Params, serialization_dir: str, recover: bool = False, semi_supervision: bool = False) -> 'TrainerPiecesForSemi':
        all_datasets = training_util.datasets_from_params(params)
        unlabelled_dataset = None 
        if semi_supervision:
            unlabelled_dataset_reader = DatasetReader.from_params(params.pop('unlabelled_dataset_reader'))
            unlabelled_data_path = params.pop('unlabelled_train_data_path')
            logger.info("Reading unlabelled training data from %s", unlabelled_data_path)
            unlabelled_dataset = unlabelled_dataset_reader.read(unlabelled_data_path)
            all_datasets['unlabelled'] = unlabelled_dataset
        else:
            for k in ['unlabelled_dataset_reader', 'unlabelled_train_data_path']:
                if k in params:
                    params.pop(k)

        datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

        if 'unlabelled' in datasets_for_vocab_creation and (not semi_supervision):
            datasets_for_vocab_creation.remove('unlabelled')

        for dataset in datasets_for_vocab_creation:
            if dataset not in all_datasets:
                raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

        logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                    ", ".join(datasets_for_vocab_creation))

        if recover and os.path.exists(os.path.join(serialization_dir, "vocabulary")):
            vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
            params.pop("vocabulary", {})
        else:
            vocab = Vocabulary.from_params(
                    params.pop("vocabulary", {}),
                    (instance for key, dataset in all_datasets.items()
                     for instance in dataset
                     if key in datasets_for_vocab_creation)
            )

        model = Model.from_params(vocab=vocab, params=params.pop('model'))

        # Initializing the model can have side effect of expanding the vocabulary
        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(model.vocab)
        validation_iterator_params = params.pop("validation_iterator", None)
        if validation_iterator_params:
            validation_iterator = DataIterator.from_params(validation_iterator_params)
            validation_iterator.index_with(model.vocab)
        else:
            validation_iterator = None

        train_data = all_datasets['train']
        validation_data = all_datasets.get('validation')
        test_data = all_datasets.get('test')

        trainer_params = params.pop("trainer")
        no_grad_regexes = trainer_params.pop("no_grad", ())
        for name, parameter in model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        frozen_parameter_names, tunable_parameter_names = \
                    get_frozen_and_tunable_parameter_names(model)
        logger.info("Following parameters are Frozen  (without gradient):")
        for name in frozen_parameter_names:
            logger.info(name)
        logger.info("Following parameters are Tunable (with gradient):")
        for name in tunable_parameter_names:
            logger.info(name)

        return TrainerPiecesForSemi(model, iterator,
                             train_data, validation_data, test_data,
                             validation_iterator, unlabelled_dataset, trainer_params)



"""
for every batch from g2, generate 'ratio' batches from g1. Assuming g1 is bigger. 
"""
def mix_generators_bm(g1,g2,g1_size, g2_size,g1_id = 0):
    if g1_size < g2_size:
        return _mix_generators(g2,g1,g2_size,g1_size,(g1_id+1)%2)
    else:
        return _mix_generators(g1,g2,g1_size,g2_size,g1_id)

def mix_generators_cm(g1, g2, id1):
    for (x1,x2) in zip(g1,g2):
        yield (x1,id1)
        yield (x2, (id1 + 1)%2)

def mix_generators_em(g1,g2,id1):
    for x in g1:
        yield(x,id1)
    #
    if g2 is not None:
        for x in g2:
            yield(x,(id1 + 1)%2)

def _mix_generators(g1,g2,g1_size,g2_size, g1_id =0):
    count = 0
    g1_active = True
    g2_active = True  

    g2_id = (g1_id + 1) %2 
    ratio  = int(round(g1_size/g2_size)) + 1
    #for every batch from g2, generate 'ratio' batches from g1
    while True:
        if (g2_active and (count % ratio == 0)):
            try:
                yield (g2.__next__(),g2_id)
            except StopIteration:
                g2_active = False
        elif g1_active:
            try:
                yield (g1.__next__(),g1_id)
            except StopIteration:
                g1_active = False
        #
        count += 1
        if not (g1_active or g2_active):
            break
    #

def get_mixer(it1, ds1, it2, ds2, num_gpus=1,id1 = 0, which_mixer='bm',min_pct_of_ds2 = 0.0):
    #Pdb().set_trace()
    num_batches1 = math.ceil(it1.get_num_batches(ds1)/num_gpus)
    if ds2 is None:
        num_batches2 = 0
    else:
        num_batches2 = math.ceil(it2.get_num_batches(ds2)/num_gpus)
    
    id2 = (id1+1)%2
    if (which_mixer in ['em','bm']) or (num_batches1 == 0) or (num_batches2 == 0):
        total_batches = num_batches1+num_batches2
        num_epochs1 = 1
        num_epochs2 = 1
    elif which_mixer == 'cm':
        total_batches = round(max(2*num_batches1, 2*min_pct_of_ds2*num_batches2))
        #total_batches = 2*max(num_batches1, num_batches2)


        num_epochs1 = round(total_batches/(2*num_batches1))
        #num_epochs1 = 1
        num_epochs2 = math.ceil(total_batches/(2*num_batches2))
    else:
        raise "incorrect value of which_mixer {}".format(which_mixer)
    raw_g1 = it1(ds1, num_epochs = num_epochs1)
    raw_g2 = it2(ds2, num_epochs = num_epochs2)
    g1 = lazy_groups_of(raw_g1, num_gpus)
    if ds2 is None:
        g2 = None
    else:
        g2 = lazy_groups_of(raw_g2, num_gpus)
    if (which_mixer == 'em') or (num_batches1 == 0) or (num_batches2 == 0):
        mixer = mix_generators_em(g1,g2,id1)
    elif which_mixer == 'bm':
        mixer = mix_generators_bm(g1,g2,num_batches1, num_batches2,id1)
    elif which_mixer == 'cm':
        mixer = mix_generators_cm(g1, g2,id1)
    #
    return (mixer, total_batches)
    
class SemiSupervisedTrainer:
    def __init__(self, trainer, constraints_model, dd_optimizer, big_iterator, unlabelled_dataset=None, semi_supervision=False,  which_mixer = 'cm', dd_warmup_iters=100, dd_update_freq=2, constraints_wt=0, calc_valid_freq=1, backprop_after_xbatches = 1, min_pct_of_unlabelled=0.0,dd_semi_warmup_iters=0.0,dd_increase_freq_after=None,dd_increase_freq_by=1, dd_decay_lr=0):
        #for dual training, need -
        #constraint model
        #dual optimizer
        #few params like - 
        #weight of each constraint
        #total weight of constraint vs supervised loss
        #warmup iterations
        self.dd_decay_lr = dd_decay_lr
        self.dd_increase_freq_by = dd_increase_freq_by
        self.dd_optimizer = dd_optimizer
        self.constraints_model = constraints_model 
        self.semi_supervision = semi_supervision 
        self.trainer = trainer
        self.unlabelled_dataset = unlabelled_dataset
        self.big_iterator = big_iterator
        self.dd_warmup_iters =  dd_warmup_iters 
        self.dd_update_freq = dd_update_freq 
        self.which_mixer = which_mixer
        self.constraints_wt = constraints_wt 
        self.calc_valid_freq = calc_valid_freq
        self.labelled_id = 0
        self.total_supervised_iters = 0.0
        self.backprop_after_xbatches = backprop_after_xbatches
        self.min_pct_of_unlabelled = min_pct_of_unlabelled 
        self.dd_semi_warmup_iters = max(self.dd_warmup_iters,dd_semi_warmup_iters)
        if not self.semi_supervision:
            self.unlabelled_dataset = None
        
        if self.constraints_wt == 0:
            self.constraints_model = None 

        self.dd_increase_freq_after = dd_increase_freq_after 
        self.count_lambda_updates = 0.0
        self.last_lambda_update = 0
        self.load_constraints_model()

    def batch_loss(self, batch_group: List[TensorDict], for_training: bool, eval_metric = True):
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self.trainer._multiple_gpu:
            output_dict = training_util.data_parallel(batch_group, self.trainer.model, self.trainer._cuda_devices)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self.trainer._cuda_devices[0])
            output_dict = self.trainer.model(**batch,eval_metric=eval_metric)

        if for_training and eval_metric:
            output_dict['regularization_penalty'] = self.trainer.model.get_regularization_penalty()  

        return output_dict 

    def step(self,loss):
        self.trainer.optimizer.zero_grad()
        loss.backward()
        batch_grad_norm = self.trainer.rescale_gradients()
        self.trainer.optimizer.step()
        return batch_grad_norm
 
    def semi_train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self.trainer._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")
    
        train_loss = 0.0
        # Set the model to "train" mode.
        self.trainer.model.train()
    
        num_gpus = len(self.trainer._cuda_devices)
        
        self.trainer._last_log = time.time()
        last_save_time = time.time()
    
        batches_this_epoch = 0
        if self.trainer._batch_num_total is None:
            self.trainer._batch_num_total = 0
    
        histogram_parameters = set(self.trainer.model.get_parameters_for_histogram_tensorboard_logging())
        #Pdb().set_trace() 
        mixed_generator, num_training_batches = get_mixer(self.trainer.iterator, self.trainer.train_data, self.trainer.iterator,  self.unlabelled_dataset,num_gpus, self.labelled_id, self.which_mixer,self.min_pct_of_unlabelled)
        #mixed_generator, num_training_batches = get_mixer(self.trainer.iterator, self.trainer.train_data, self.trainer._validation_iterator,  self.unlabelled_dataset,num_gpus, self.labelled_id, self.which_mixer)
        
        #generator for lambda update
        mixed_generator_for_lambda, _ = get_mixer(self.trainer.iterator, self.trainer.train_data, self.trainer.iterator,  self.unlabelled_dataset, num_gpus, self.labelled_id, 'cm',1.0)
        #mixed_generator_for_lambda, _ = get_mixer(self.trainer._validation_iterator, self.trainer.train_data, self.trainer._validation_iterator,  self.unlabelled_dataset, num_gpus, self.labelled_id, 'cm')

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(mixed_generator,
                                         total=num_training_batches)
        #train_generator_tqdm = Tqdm.tqdm(zip(train_generator,unlabelled_train_generator),
        #                                 total=num_training_batches)
        cumulative_batch_size = 0
        unlabelled_loss = 0
        unlabelled_batches_this_epoch = 0
   
        batches_since_last_step = 0 
        agg_loss = 0.0
        flag = False 
        batch_grad_norm = None 
        for batch_group,group_id in train_generator_tqdm:
            #print(batch_group[0]['sentence']['tokens'].shape)
            if self.total_supervised_iters < self.dd_semi_warmup_iters and group_id != self.labelled_id:
                continue
            output_dict = self.batch_loss(batch_group, for_training=True,eval_metric = (group_id == self.labelled_id))
            penalties = defaultdict(float)

            if self.constraints_model is not None:
                penalties = self.constraints_model(output_dict['task1_tag_logits'], output_dict['task2_tag_logits'], output_dict['mask']) 

            loss = 0.0
            if 'loss' in output_dict:
                loss = output_dict['loss']
                train_loss += loss.item() 
            loss += output_dict.get('regularization_penalty',0.0)
            
            loss += self.constraints_wt*penalties['loss']
            
            unlabelled_loss += penalties['loss'].item() if torch.is_tensor(penalties['loss']) else penalties['loss']
            
            agg_loss += loss
            batches_since_last_step += 1
            
            if batches_since_last_step == self.backprop_after_xbatches:
                #print("STEP THROUGH! : {}. loss: {} agg_loss: {}".format(group_id, loss, agg_loss))
                batch_grad_norm = self.step(agg_loss)
                batches_since_last_step = 0
                agg_loss = 0.0
                flag = False 
            else:
                flag = True 
                #print("skipp : {}. loss: {} agg_loss: {}".format(group_id, loss, agg_loss))

            if (group_id != self.labelled_id):
                unlabelled_batches_this_epoch += 1
                #self.trainer.optimizer.zero_grad()
                #loss.backward()
                #batch_grad_norm = self.trainer.rescale_gradients()
                #self.trainer.optimizer.step()
            else:
                self.total_supervised_iters += 1.0
                batches_this_epoch += 1
                self.trainer._batch_num_total += 1
                batch_num_total = self.trainer._batch_num_total
                 
                #self.trainer.optimizer.zero_grad()
                #loss.backward()
                #batch_grad_norm = self.trainer.rescale_gradients()
    
                # This does nothing if batch_num_total is None or you are using an
                # LRScheduler which doesn't update per batch.
                if self.trainer._learning_rate_scheduler:
                    self.trainer._learning_rate_scheduler.step_batch(batch_num_total)
    
                if self.trainer._tensorboard.should_log_histograms_this_batch():
                    # get the magnitude of parameter updates for logging
                    # We need a copy of current parameters to compute magnitude of updates,
                    # and copy them to CPU so large models won't go OOM on the GPU.
                    param_updates = {name: param.detach().cpu().clone()
                                     for name, param in self.trainer.model.named_parameters()}
                    #self.trainer.optimizer.step()
                    for name, param in self.trainer.model.named_parameters():
                        param_updates[name].sub_(param.detach().cpu())
                        update_norm = torch.norm(param_updates[name].view(-1, ))
                        param_norm = torch.norm(param.view(-1, )).cpu()
                        self.trainer._tensorboard.add_train_scalar("gradient_update/" + name,
                                                           update_norm / (param_norm + 1e-7))
                else:
                    pass 
                    #self.trainer.optimizer.step()
    
                # Update moving averages
                if self.trainer._moving_average is not None:
                    self.trainer._moving_average.apply(batch_num_total)
            #
                metrics = training_util.get_metrics(self.trainer.model, train_loss, batches_this_epoch)
                metrics["uloss"] = float(unlabelled_loss / (batches_this_epoch+unlabelled_batches_this_epoch)) 
                # Update the description with the latest metrics
                description = training_util.description_from_metrics(metrics)
                train_generator_tqdm.set_description(description, refresh=False)
        
                # Log parameter values to Tensorboard
                if self.trainer._tensorboard.should_log_this_batch() and batch_grad_norm is not None:
                    self.trainer._tensorboard.log_parameter_and_gradient_statistics(self.trainer.model, batch_grad_norm)
                    self.trainer._tensorboard.log_learning_rates(self.trainer.model, self.trainer.optimizer)
        
                    self.trainer._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                    self.trainer._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})
        
                if self.trainer._tensorboard.should_log_histograms_this_batch():
                    self.trainer._tensorboard.log_histograms(self.trainer.model, histogram_parameters)
        
                if self.trainer._log_batch_size_period:
                    cur_batch = sum([training_util.get_batch_size(batch) for batch in batch_group])
                    cumulative_batch_size += cur_batch
                    if (batches_this_epoch - 1) % self.trainer._log_batch_size_period == 0:
                        average = cumulative_batch_size/batches_this_epoch
                        logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                        self.trainer._tensorboard.add_train_scalar("current_batch_size", cur_batch)
                        self.trainer._tensorboard.add_train_scalar("mean_batch_size", average)
        
                # Save model if needed.
                if self.trainer._model_save_interval is not None and (
                        time.time() - last_save_time > self.trainer._model_save_interval
                ):
                    last_save_time = time.time()
                    self.trainer._save_checkpoint(
                            '{0}.{1}'.format(epoch, training_util.time_to_str(int(last_save_time)))
                    )
    
            #lambda update
            #if  (self.constraints_model is not None) and (self.dd_optimizer is not None) and (self.total_supervised_iters >= self.dd_warmup_iters) and (batches_this_epoch % self.dd_update_freq == 0):
            if  (self.constraints_model is not None) and (self.dd_optimizer is not None) and (self.total_supervised_iters >= self.dd_warmup_iters) and (self.total_supervised_iters - self.last_lambda_update >=  self.dd_update_freq):
                for batch_group,group_id in mixed_generator_for_lambda:
                    self.lambda_update(batch_group)        
                    self.last_lambda_update = self.total_supervised_iters 
                    break
        
                self.count_lambda_updates += 1
                if (self.dd_increase_freq_after is not None) and (self.count_lambda_updates % self.dd_increase_freq_after == 0):
                    self.dd_update_freq += self.dd_increase_freq_by 
        if flag:
            batch_grad_norm = self.step(agg_loss)
            batches_since_last_step = 0
            agg_loss = 0.0
            flag = False 

        
        #lambda update
        #if (self.constraints_model is not None) and (self.dd_optimizer is not None) and (self.total_supervised_iters >= self.dd_warmup_iters):
        if (self.constraints_model is not None) and (self.dd_optimizer is not None) and (self.total_supervised_iters >= self.dd_warmup_iters) and (self.total_supervised_iters - self.last_lambda_update >= self.dd_update_freq):
            for batch_group,group_id in mixed_generator_for_lambda:
                self.lambda_update(batch_group)        
                self.last_lambda_update = self.total_supervised_iters 
                break

            self.count_lambda_updates += 1
            if (self.dd_increase_freq_after is not None) and (self.count_lambda_updates % self.dd_increase_freq_after == 0):
                self.dd_update_freq += self.dd_increase_freq_by 
        
        metrics = training_util.get_metrics(self.trainer.model, train_loss, batches_this_epoch, reset=True)
        metrics['cpu_memory_MB'] = peak_cpu_usage
        metrics['lb'] = batches_this_epoch
        metrics['ub'] = unlabelled_batches_this_epoch 
        metrics["uloss"] = float(unlabelled_loss / (batches_this_epoch+unlabelled_batches_this_epoch)) 
        if self.constraints_model is not None:
            lambda_stats_dict = self.constraints_model.lambda_stats()
            metrics.update(lambda_stats_dict)
        for (gpu_num, memory) in gpu_usage:
            metrics['gpu_'+str(gpu_num)+'_memory_MB'] = memory
        return metrics

    def lambda_update(self, batch_group):
        with torch.no_grad():
            self.trainer.model.eval()
            output_dict = self.batch_loss(batch_group, for_training=True,eval_metric = False)
            self.trainer.model.train()
        #
        penalties = self.constraints_model(output_dict['task1_tag_logits'].detach(), output_dict['task2_tag_logits'].detach(), output_dict['mask'].detach()) 
        self.dd_optimizer.zero_grad()
        loss = -1.0*penalties['loss']
        loss.backward()
        self.dd_optimizer.step()
        #Pdb().set_trace()
        if self.dd_decay_lr:
            for param_group in self.dd_optimizer.param_groups:
                param_group['lr'] = param_group['lr']*math.sqrt((self.count_lambda_updates+1)/(self.count_lambda_updates+2))
        if self.total_supervised_iters % 1000 == 0:
            self.save_constraints_model()

    def custom_train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self.trainer._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")
    
        training_util.enable_gradient_clipping(self.trainer.model, self.trainer._grad_clipping)
    
        logger.info("Beginning training.")
    
        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()
    
        metrics['best_epoch'] = self.trainer._metric_tracker.best_epoch
        for key, value in self.trainer._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value
    
        for epoch in range(epoch_counter, self.trainer._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self.semi_train_epoch(epoch)
    
            # get peak of memory usage
            if 'cpu_memory_MB' in train_metrics:
                metrics['peak_cpu_memory_MB'] = max(metrics.get('peak_cpu_memory_MB', 0),
                                                    train_metrics['cpu_memory_MB'])
            for key, value in train_metrics.items():
                if key.startswith('gpu_'):
                    metrics["peak_"+key] = max(metrics.get("peak_"+key, 0), value)
    
            """
            if self.unlabelled_dataset is not None:
                unlabelled_metrics = unlabelled_train_epoch(self.trainer, self.unlabelled_dataset, epoch)
                for key, value in unlabelled_metrics.items():
                    if key.startswith('gpu_'):
                        metrics["peak_"+'un_'+key] = max(unlabelled_metrics.get("peak_"+key, 0), value)
                    else:
                        metrics['un_'+key] = value
            """
    
            #if self.trainer._validation_data is not None:
            if self.trainer._validation_data is not None and ((epoch - epoch_counter) % self.calc_valid_freq == (self.calc_valid_freq-1)):
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self.trainer._validation_loss()
                    val_metrics = training_util.get_metrics(self.trainer.model, val_loss, num_batches, reset=True)
    
                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self.trainer._validation_metric]
                    self.trainer._metric_tracker.add_metric(this_epoch_val_metric)
    
                    if self.trainer._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break
    
            self.trainer._tensorboard.log_metrics(train_metrics, val_metrics=val_metrics, log_to_console=True)
    
            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = time.strftime("%H:%M:%S", time.gmtime(training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch
    
            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value
    
            is_best_so_far = False
            if self.trainer._metric_tracker.is_best_so_far():
                is_best_so_far = True 
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value
    
                self.trainer._metric_tracker.best_epoch_metrics = val_metrics
    
            if self.trainer._serialization_dir:
                dump_metrics(os.path.join(self.trainer._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)
   
            #Pdb().set_trace()
            if self.trainer._learning_rate_scheduler:
                # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
                # if it doesn't, the validation metric passed here is ignored.
                self.trainer._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
    
            self.trainer._save_checkpoint(epoch)
            if self.constraints_model is not None:
                spath = self.save_constraints_model(epoch)   
                if is_best_so_far:
                    shutil.copyfile(spath,os.path.join(self.trainer._serialization_dir,'best_dd_checkpoint.pth'))
    
            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))
    
            if epoch < self.trainer._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self.trainer._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)
            
            self.trainer.model.train()
            epochs_trained += 1
    
        # Load the best model state before returning
        best_model_state = self.trainer._checkpointer.best_model_state()
        if best_model_state:
            self.trainer.model.load_state_dict(best_model_state)
    
        return metrics
    
    def save_constraints_model(self,epoch=-1):
        chpoint = {}
        chpoint['epoch'] = epoch
        chpoint['num_iters'] = self.total_supervised_iters 
        chpoint['constraints_model'] = self.constraints_model.state_dict()
        chpoint['dd_optim'] = self.dd_optimizer.state_dict()
        chpoint['dd_update_freq'] = self.dd_update_freq 

        torch.save(chpoint, os.path.join(self.trainer._serialization_dir, 'dd_checkpoint.pth'))
        return os.path.join(self.trainer._serialization_dir, 'dd_checkpoint.pth')

    def load_constraints_model(self):
        chfile = os.path.join(self.trainer._serialization_dir, 'dd_checkpoint.pth')
        if os.path.exists(chfile):
            cp = torch.load(chfile)
            if self.dd_optimizer is not None:
                self.dd_optimizer.load_state_dict(cp['dd_optim'])
            if self.constraints_model is not None:
                self.constraints_model.load_state_dict(cp['constraints_model'])
            self.total_supervised_iters = cp['num_iters']
            self.dd_update_freq = cp['dd_update_freq']
        #print(self.constraints_model.lambda_list[0])


