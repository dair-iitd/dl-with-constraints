
from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, ArrayField
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
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

import pickle


@DatasetReader.register("mtl_reader")
class JointSeq2SeqDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 how_many_sentences=None, start_from=0,
                 return_labels=True) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}
        self.how_many_sentences = how_many_sentences
        self.start_from = start_from
        self.return_labels = return_labels

    def text_to_instance(self, idx: int,
                         tokens: List[Token],
                         task1_tags: List[str] = None,
                         task2_tags: List[str] = None
                         ) -> Instance:

        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if task1_tags:
            task1_label_field = SequenceLabelField(labels=task1_tags,
                                                   sequence_field=sentence_field,
                                                   label_namespace='task1_labels')

            fields["task1_labels"] = task1_label_field
            task2_label_field = SequenceLabelField(labels=task2_tags,
                                                   sequence_field=sentence_field,
                                                   label_namespace='task2_labels')

            fields["task2_labels"] = task2_label_field

        fields["idx"] = ArrayField(np.array(idx))
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        sentences = pickle.load(open(file_path, 'rb'))
        sentences = sentences[self.start_from:]
        if self.how_many_sentences is None:
            self.how_many_sentences = len(sentences)
        #
        self.how_many_sentences = min(self.how_many_sentences, len(sentences))

        for i, this_sentence in enumerate(sentences[:self.how_many_sentences]):
            # sentence is a list of triplets - (word, mer_tag, men_tag)
            sentence, task1_tags, task2_tags = zip(*(this_sentence))
            if self.return_labels:
                yield self.text_to_instance(i, [Token(word.lower()) for word in sentence], task1_tags, task2_tags)
            else:
                yield self.text_to_instance(i, [Token(word.lower()) for word in sentence], None, None)
