from abc import abstractmethod
import six
import abc

@six.add_metaclass(abc.ABCMeta)
class BasePenalty():
    """ 
    This is an abstract class for any constraint 
    """

    def __init__(self):
        super(BasePenalty,self).__init__()

    @abstractmethod
    def num_constraints(self):
        pass

    @abstractmethod
    def get_penalty(self,scores,indicator):
        pass

