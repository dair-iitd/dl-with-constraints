
from IPython.core.debugger import Pdb
from typing import Iterator, List, Dict,Optional
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
import torch.nn.functional as F
import utils
import settings
import logging
from allennlp.nn import InitializerApplicator, RegularizerApplicator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
@Model.register("mtl_tagger")
class LstmTagger(Model):
    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 mlp1: List[int],
                 mlp2: List[int],
                 task1_ignore_classes: List[str] = None,
                 task2_ignore_classes: List[str] = None,
                 task1_wts_uniform=False,
                 task2_wts_uniform=False,
                 task1_metric = 'f1',
                 task2_metric= 'f1',
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        self.task1_hidden2tag = torch.nn.Sequential(
            *self.create_mlp(mlp1, vocab, 'task1_labels'))
        self.task2_hidden2tag = torch.nn.Sequential(
            *self.create_mlp(mlp2, vocab, 'task2_labels'))

        
        self.task1_wts = torch.ones(vocab.get_vocab_size('task1_labels'))
        self.task2_wts = torch.ones(vocab.get_vocab_size('task2_labels'))
        if not task1_wts_uniform:
            self.task1_wts = utils.get_weights(vocab,'task1_labels')
        if not task2_wts_uniform:
            self.task2_wts = utils.get_weights(vocab, 'task2_labels')

        
        self.task1_accuracy = CategoricalAccuracy()
        self.task2_accuracy = CategoricalAccuracy()

        self.task1_metric_type = task1_metric
        if task1_metric == 'f1':
            self.task1_metric = SpanBasedF1Measure(
            vocab, tag_namespace="task1_labels", ignore_classes=task1_ignore_classes)
        else:
            self.task1_metric = CategoricalAccuracy()
        
        self.task2_metric_type = task2_metric
        if task2_metric == 'f1':
            self.task2_metric  = SpanBasedF1Measure(
            vocab, tag_namespace="task2_labels", ignore_classes=task2_ignore_classes)
        else:
            self.task2_metric = CategoricalAccuracy()
            

        if settings.cuda:
            self.task1_wts = self.task1_wts.cuda()
            self.task2_wts = self.task2_wts.cuda()


    def create_mlp(self, mlp1, vocab, task):
        mlp_list1 = []
        input_neurons = self.encoder.get_output_dim()
        for num_neurons, dropout in mlp1:
            mlp_list1.append(torch.nn.Linear(
                in_features=input_neurons, out_features=num_neurons))

            if dropout > 0:
                mlp_list1.append(torch.nn.Dropout(dropout))

            mlp_list1.append(torch.nn.ReLU())
            input_neurons = num_neurons 
            
        #
        mlp_list1.append(torch.nn.Linear(in_features=input_neurons,
                                         out_features=vocab.get_vocab_size(task)))

        return mlp_list1

    def forward_x(self, encoder_out, hidden2tag, labels, mask, wts, accuracy_metric, task_metric, task, eval_metric):
        tag_logits = hidden2tag(encoder_out)
        #class_probabilites = F.softmax(tag_logits, dim=-1)
        output = {task+"_tag_logits": tag_logits}

        if labels is not None:
            if eval_metric:
                accuracy_metric(tag_logits, labels, mask)
                task_metric(tag_logits, labels, mask)
            output[task+"_loss"] = sequence_cross_entropy_with_logits(
                tag_logits, labels, weights=wts)
            # , average="token")
        #
        return output

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                idx: torch.Tensor  = None,
                task1_labels: torch.Tensor = None,
                task2_labels: torch.Tensor = None,
                eval_metric: bool = True 
                ) -> Dict[str, torch.Tensor]:

        # keys of the sentence dictionary should be exactly same as that of self.text_field_embedder

        mask = get_text_field_mask(sentence)
        embeddings = self.text_field_embedder(sentence)
        encoder_out = self.encoder(embeddings, mask)

        task1_wts = None
        if task1_labels is not None:
            task1_wts = self.task1_wts[task1_labels]
            task1_wts[mask == 0] = 0
            #if settings.cuda:
            #    wts = wts.cuda()
        task2_wts = None
        if task2_labels is not None:
            task2_wts = self.task2_wts[task2_labels]
            task2_wts[mask == 0] = 0


        task1_output = self.forward_x(encoder_out, self.task1_hidden2tag,
                                      task1_labels, mask, task1_wts, self.task1_accuracy,
                                      self.task1_metric, 'task1', eval_metric)

        output = self.forward_x(encoder_out, self.task2_hidden2tag,
                                task2_labels, mask, task2_wts, self.task2_accuracy,
                                self.task2_metric,  'task2', eval_metric)

        output.update(task1_output)

        #output = task1_output
        if 'task1_loss' in output:
            #output['loss'] = output['task1_loss'] + output['task2_loss']
            #output['loss'] = output['task1_loss']
            output['loss'] = output['task1_loss']
        
        if 'task2_loss' in output:
            output['loss'] += output['task2_loss']

        output['mask'] =  mask 
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        rv = {}
        rv['t1a'] = self.task1_accuracy.get_metric(reset)
        rv['t2a'] = self.task2_accuracy.get_metric(reset)
        task1_metric = 1.0 - rv['t1a']
        task2_metric = 1.0 - rv['t2a']
        
        if self.task1_metric_type == 'f1':
            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            metric_dict_task1 = self.task1_metric.get_metric(reset=reset)
            rv.update({"t1_"+x[0]: y for x, y in metric_dict_task1.items()
                  if "overall" in x})
            #
            task1_metric = 1.0 - rv['t1_f']

        if self.task2_metric_type == 'f1':
            metric_dict_task2 = self.task2_metric.get_metric(reset=reset)
            rv.update({"t2_"+x[0]: y for x, y in metric_dict_task2.items()
               if "overall" in x})
            #
            task2_metric = 1.0 - rv['t2_f']
        
        rv['nm'] = 0.5*(task1_metric  + task2_metric)
        
        return rv

