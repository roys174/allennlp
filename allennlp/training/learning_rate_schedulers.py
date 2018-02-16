"""
AllenNLP just uses
`PyTorch learning rate schedulers <http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate>`_,
with a thin wrapper to allow registering them and instantiating them ``from_params``.

The available learning rate schedulers are

* `"step" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.StepLR>`_
* `"multi_step" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.MultiStepLR>`_
* `"exponential" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ExponentialLR>`_
* `"reduce_on_plateau" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau>`_
"""

import torch
import torch.optim.lr_scheduler
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable


class LearningRateScheduler(Registrable):
    """
    This class just allows us to implement ``Registrable`` for Pytorch :class:`LRSchedulers`.
    """
    @classmethod
    def from_params(cls, optimizer: torch.optim.Optimizer, params: Params):
        scheduler = params.pop_choice("type", LearningRateScheduler.list_available())
        return LearningRateScheduler.by_name(scheduler)(optimizer, **params.as_dict())  # type: ignore


@LearningRateScheduler.register("frozen_rate_scheduler")
class FrozenRateScheduler(LearningRateScheduler, torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs_to_freeze, groups_to_freeze,
                 unfreeze_ratio):
        self._unfrozen = False
        self.n_epochs = 0
        self.num_epochs_to_freeze = num_epochs_to_freeze
        self.groups_to_freeze = groups_to_freeze
        self.unfreeze_ratio = unfreeze_ratio

        super(FrozenRateScheduler, self).__init__(optimizer)

        # set the initial parameter groups lr to 0
        for k, param_group in enumerate(self.optimizer.param_groups):
            if k in self.groups_to_freeze:
                param_group['lr'] = 0.0

    def step(self, *args, **kwargs):
        if self._unfrozen:
            return

        self.n_epochs += 1
        # step is called once in the constructor, so unfreeze when
        # self.n_epochs == self.num_epochs_to_freeze + 1
        if self.n_epochs <= self.num_epochs_to_freeze:
            return

        # else time to unfreeze the lr
        # first get the unfrozen lr
        lrs = []
        for k, param_group in enumerate(self.optimizer.param_groups):
            if k not in self.groups_to_freeze:
                lrs.append(param_group['lr'])
        # TODO: fix this
        assert len(lrs) == 1
    
        unfrozen_lr = lrs[0] * self.unfreeze_ratio
        for k, param_group in enumerate(self.optimizer.param_groups):
            if k in self.groups_to_freeze:
                param_group['lr'] = unfrozen_lr


@LearningRateScheduler.register("learning_rate_scheduler_list")
class LearningRateSchedulerList(LearningRateScheduler):
    def __init__(self, optimizer, learning_rate_scheduler_params):
        self.lr_schedulers = [
                LearningRateScheduler.from_params(optimizer, Params(params))
                for params in learning_rate_scheduler_params
        ]

    def step(self, *args, **kwargs):
        for scheduler in self.lr_schedulers:
            scheduler.step(*args, **kwargs)



# We just use the Pytorch LRSchedulers, so here we force them into
# Registry._registry so we can build them from params.
Registrable._registry[LearningRateScheduler] = {   # pylint: disable=protected-access
        "step": torch.optim.lr_scheduler.StepLR,
        "multi_step": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "frozen_rate_scheduler": FrozenRateScheduler,
        "learning_rate_scheduler_list": LearningRateSchedulerList
}
