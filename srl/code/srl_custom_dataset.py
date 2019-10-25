from IPython.core.debugger import Pdb
import logging
from typing import Dict, List, Iterable

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence
from allennlp.data.dataset_readers.dataset_utils.span_utils import ( to_bioul, bioul_tags_to_spans )
from copy import deepcopy
import random
import os
from nltk.tree import Tree
from torch import Tensor
import numpy as np
import pickle
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



def leftMost(inTree):
    counter = 0
    def _leftMost(inTree): 
        nonlocal counter
        if type(inTree) is Tree and len(inTree) > 0: 
            list_children = [] 
            for child in inTree: 
                list_children.append(_leftMost(child)) 
            cur_label = None 
            try: 
                cur_label = list_children[0].label() 
            except: 
                cur_label = list_children[0] 
            return Tree(cur_label, list_children) 
        else:
            counter += 1
            return (counter-1)

    return _leftMost(inTree)

def rightMost(inTree):
    counter = 0
    def _rightMost(inTree):
        nonlocal counter
        if type(inTree) is Tree and len(inTree) > 0: 
            list_children = [] 
            for child in inTree: 
                list_children.append(_rightMost(child)) 
            cur_label = None 
            try: 
                cur_label = list_children[-1].label() 
            except: 
                cur_label = list_children[-1] 
            return Tree(cur_label, list_children) 
        else: 
            counter += 1
            return (counter-1)

    return _rightMost(inTree)

# Check - 


# def get_leaves(inTree, leaf_list): 
#     if type(inTree) is Tree: 
#         for child in inTree: 
#             get_leaves(child, leaf_list) 
#     else: 
#         leaf_list.append(inTree)

def addToList(inTree, numList):
    if type(inTree) is Tree:
        num = inTree.label()
        numList.append(num)
        for child in inTree:
            addToList(child, numList)
    else:
        numList.append(inTree)

@DatasetReader.register("srl_custom")
class CustomSrlReader(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for semantic role labelling. It returns a dataset of instances with the
    following fields:

    tokens : ``TextField``
        The tokens in the sentence.
    verb_indicator : ``SequenceLabelField``
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given verb in a BIO format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.

    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 label_encoding: str = "BIO",
                 percent_data: int = 100,
                 random_data_seed: int = 42,
                 return_labels: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier
        self.label_encoding = label_encoding
        self.percent_data = percent_data
        self.random_data_seed = random_data_seed
        self.return_labels = return_labels
        print(f"Dataset reader label encoding: {self.label_encoding}")
        print(f"Percent data: {self.percent_data}")


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        #Pdb().set_trace()
        data_split = os.path.basename(os.path.normpath(file_path))
        
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)

        # Set random seed if percent is not 100
        if(self.percent_data < 100):
            random.seed(self.random_data_seed)

        # Write sentence, parse tree, span matrix to file
        # fout = open(f"srl_spans_{data_split}.pkl", "wb")

        print(f"return_labels: {self.return_labels}")

        for sentence in self._ontonotes_subset(ontonotes_reader, file_path, self._domain_identifier):
            if(self.percent_data < 100 and data_split == "train"):
                select_data = random.randint(1, 101)
                if(select_data > self.percent_data):
                    continue
            tokens = [Token(t) for t in sentence.words]
            parseTree = sentence.parse_tree

            # Convert tree to span list            

            if not sentence.srl_frames:         
                # Sentence contains no predicates.                            
                verb_label = [0 for _ in tokens]
                if self.return_labels:
                    tags = ["O" for _ in tokens]                    
                    yield self.text_to_instance(tokens, verb_label, parseTree, tags)
                else:
                    yield self.text_to_instance(tokens, verb_label, parseTree, None)
            else:                        
                for (_, tags) in sentence.srl_frames:                    
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    if self.return_labels:                        
                        yield self.text_to_instance(tokens, verb_indicator, parseTree, tags)
                    else:
                        yield self.text_to_instance(tokens, verb_indicator, parseTree, None)

        # fout.close()

    @staticmethod
    def _ontonotes_subset(ontonotes_reader: Ontonotes,
                          file_path: str,
                          domain_identifier: str) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):            
            if domain_identifier is None or f"/{domain_identifier}/" in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         parseTree: Tree,
                         tags: List[str] = None,
                         fout=None
                         ) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ

        # Convert tags to BIOUL QUESTION -  BIO or IOB1?
        # print(f"Tags before: {tags}")



        if(self.label_encoding == "BIOUL"):
            if(tags is not None):
                old_tags = deepcopy(tags)
                tags = to_bioul(tags, encoding = "BIO")
                try:
                    spans = bioul_tags_to_spans(tags)
                except InvalidTagSequence:
                    print(f"Old tags: {old_tags}")
                    print(f"New tags: {tags}\n")

            # Create span matrix from parse tree
            leftLabelsTree = leftMost(parseTree)
            rightLabelsTree = rightMost(parseTree)
            
            # leaves = []
            # right_leaves = []
            # get_leaves(parseTree, leaves)
            # get_leaves(parseTree, right_leaves)
            # assert(leaves == right_leaves)
            # leaf2idx = {}
            # for idx, leaf in enumerate(leaves):
            #     leaf2idx[leaf] = idx

            leftList = []
            rightList = []

            addToList(leftLabelsTree, leftList)
            addToList(rightLabelsTree, rightList)

            if len(leftList) != len(rightList):
                raise Exception(f"For tree {parseTree}, leftList and rightList lengths do not match")

            span_matrix = np.zeros([len(tokens), len(tokens)])

            for idx in range(len(leftList)):
                leftLabel, rightLabel = leftList[idx], rightList[idx]
                if(leftLabel == rightLabel):
                    continue
                span_matrix[leftLabel, rightLabel] = 1




        # print(f"Tags after: {tags}\n")

        # print(tokens)
        # print(verb_label)
        # print(tags)

        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, text_field)
        if(self.label_encoding == "BIOUL"):
            fields['span_matrix'] = ArrayField(span_matrix)

        if all([x == 0 for x in verb_label]):
            verb = None
        else:
            verb = tokens[verb_label.index(1)].text
        metadata_dict = {"words": [x.text for x in tokens],
                         "verb": verb}
        if tags:
            fields['tags'] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags
        fields["metadata"] = MetadataField(metadata_dict)

        if(fout is not None):
            srl_dict = {
                            "parse_tree": parseTree,
                            "span_matrix": span_matrix
                       }
            pickle.dump(srl_dict, fout)

        return Instance(fields)
