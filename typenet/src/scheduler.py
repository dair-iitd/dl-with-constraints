
from torch.optim import lr_scheduler

#modified from - https://pytorch.org/docs/0.3.1/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau

class FlagOnPlateau():
    def __init__(self, mode='min', patience=3,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 ):

        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = 0
        self.num_of_plateaus = 0
        self.on_plateau = False

        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)

        self._init_getThresholdFn(self.mode,self.threshold,self.threshold_mode)
        self.reset()	

    def reset(self):
        self.best = self.mode_worse
        self.num_bad_epochs = 0


    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1

        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            self.on_plateau = False 
        else:
            self.num_bad_epochs += 1
        
        if (not self.on_plateau) and (self.num_bad_epochs > self.patience):
            self.on_plateau = True
            self.num_of_plateaus += 1
        if self.verbose:
            print('best: {}, current: {}, thr: {}, #iters: {}, #baditers:{}, #patience: {}, #pu:{}, onpu:{} '.format(self.best, current, self.getThresholdFn(self.best), self.last_epoch, self.num_bad_epochs,self.patience, self.num_of_plateaus, self.on_plateau))

    def has_plateaued(self):
        return (self.num_of_plateaus > 0)
    
    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + mode + ' is unknown!')
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
            self.mode_worse = float('Inf')
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
            self.mode_worse = float('Inf')
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.is_better = lambda a, best: a > best * rel_epsilon
            self.mode_worse = -float('Inf')
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold
            self.mode_worse = -float('Inf')


    def _init_getThresholdFn(self, mode, threshold, threshold_mode):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.getThresholdFn = lambda best: (best * rel_epsilon)
        elif mode == 'min' and threshold_mode == 'abs':
            self.getThresholdFn = lambda best: best - threshold
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.getThresholdFn = lambda best: best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.getThresholdFn = lambda best: best + threshold




class CustomReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):
    def __init__(self,optimizer,kwargs={}, maxPatienceToStopTraining=20):
        super(CustomReduceLROnPlateau, self).__init__(optimizer,**kwargs)
        self.unconstrainedBadEpochs = self.num_bad_epochs
        self.maxPatienceToStopTraining = maxPatienceToStopTraining
        self._init_getThresholdFn(self.mode,self.threshold,self.threshold_mode)

    def _init_getThresholdFn(self, mode, threshold, threshold_mode):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.getThresholdFn = lambda best: (best * rel_epsilon)
        elif mode == 'min' and threshold_mode == 'abs':
            self.getThresholdFn = lambda best: best - threshold
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.getThresholdFn = lambda best: best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.getThresholdFn = lambda best: best + threshold



    def step(self, metrics, epoch=None):
        if self.is_better(metrics, self.best):
            self.unconstrainedBadEpochs = 0 
        else:
            self.unconstrainedBadEpochs += 1
        #   
        super(CustomReduceLROnPlateau,self).step(metrics,epoch)


    def shouldStopTraining(self):
        self.currentThreshold = self.getThresholdFn(self.best)
        print("Num_bad_epochs: {0}, unconstrainedBadEpochs: {1}, bestMetric: {2}, currentThreshold: {3}".format(self.num_bad_epochs, self.unconstrainedBadEpochs, self.best,self.currentThreshold))
        return(self.unconstrainedBadEpochs > self.maxPatienceToStopTraining)
    


