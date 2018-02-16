# pylint: disable=invalid-name
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.learning_rate_schedulers import FrozenRateScheduler, \
        LearningRateSchedulerList, LearningRateScheduler
from allennlp.common.params import Params


class TestFrozenRateScheduler(AllenNlpTestCase):
    def test_frozen_rate_scheduler(self):
        param_groups = [{'params': [torch.nn.Parameter(torch.rand(5)), 
                                    torch.nn.Parameter(torch.rand(3))],
                            'lr': 0.1},
                        {'params': [torch.nn.Parameter(torch.rand(2))]}]
        optim = torch.optim.Adam(param_groups)
        frozen_rate_scheduler = FrozenRateScheduler(optim, 5, [0], 0.1)

        assert optim.param_groups[0]['lr'] == 0
        assert optim.param_groups[1]['lr'] == 0.001

        for k in range(4):
            frozen_rate_scheduler.step()

        assert optim.param_groups[0]['lr'] == 0
        assert optim.param_groups[1]['lr'] == 0.001

        frozen_rate_scheduler.step()

        assert optim.param_groups[0]['lr'] == 0.0001
        assert optim.param_groups[1]['lr'] == 0.001

        frozen_rate_scheduler.step()

        assert optim.param_groups[0]['lr'] == 0.0001
        assert optim.param_groups[1]['lr'] == 0.001


class TestLearningRateSchedulerList(AllenNlpTestCase):
    def test_learning_rate_scheduler_list(self):
        param_groups = [{'params': [torch.nn.Parameter(torch.rand(5)),
                                    torch.nn.Parameter(torch.rand(3))],
                            'lr': 0.1},
                        {'params': [torch.nn.Parameter(torch.rand(2))]}]
        optim = torch.optim.Adam(param_groups)

        params = Params(
            {"type": "learning_rate_scheduler_list",
             "learning_rate_scheduler_params": [
                {"type": "reduce_on_plateau", "mode": "max",
                 "factor": 0.5, "patience": 0},
                {"type": "frozen_rate_scheduler", "num_epochs_to_freeze": 5,
                 "groups_to_freeze": [0], "unfreeze_ratio": 0.1}
            ]}
        )

        lr_scheduler = LearningRateScheduler.from_params(optim, params)

        assert optim.param_groups[0]['lr'] == 0
        assert optim.param_groups[1]['lr'] == 0.001

        # update for 3 epochs with increasing metric, then one with decreasing
        for k in range(3):
            lr_scheduler.step(k)
        lr_scheduler.step(1)

        assert optim.param_groups[0]['lr'] == 0
        assert optim.param_groups[1]['lr'] == 0.001 * 0.5

        # this step unfreeze group 0 weights
        lr_scheduler.step(10)

        assert optim.param_groups[0]['lr'] == 0.001 * 0.5 * 0.1
        assert optim.param_groups[1]['lr'] == 0.001 * 0.5

        lr_scheduler.step(11)
        lr_scheduler.step(12)
        lr_scheduler.step(7)
        lr_scheduler.step(15)

        assert optim.param_groups[0]['lr'] == 0.001 * 0.5 * 0.1 * 0.5
        assert optim.param_groups[1]['lr'] == 0.001 * 0.5 * 0.5

