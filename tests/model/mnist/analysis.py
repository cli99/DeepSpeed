from pkgutil import get_loader
from smdebug.rules import Rule, invoke_rule
from smdebug.trials import create_trial
import numpy as np
import matplotlib.pyplot as plt
import collections

import torch
from smdebug.rules.action import Actions
from smdebug.core.logger import get_logger
import logging
from smdebug.exceptions import NoMoreData

smdebug_dir = "./db"

logger = get_logger()
logger.setLevel(logging.INFO)


class WeightRatio(Rule):
    def __init__(self, base_trial, action_str=None):
        super().__init__(base_trial, action_str)
        self.tensors = collections.OrderedDict()

    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(regex=".*weight"):
            if "gradient" not in tname:
                if "lstm" in tname:
                    try:
                        tensor = self.base_trial.tensor(tname).value(step)
                        if tname not in self.tensors:
                            self.tensors[tname] = {}

                        # st.write(f" Tensor  {tname}  has weights with variance: {np.var(tensor.flatten())} ")
                        self.tensors[tname][step] = tensor
                    except:
                        self.logger.warning(f"Can not fetch tensor {tname}")
                        # st.write(f"Can not fetch tensor {tname}")
        return False


class WeightsNanRule(Rule):
    def __init__(self, base_trial, action_str=None):
        super().__init__(base_trial, action_str)

    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(collection="weights"):
            t = self.base_trial.tensor(tname)
            if torch.isnan(t):
                return True
        return False


class WeightsinitializationRule(Rule):
    def __init__(self, base_trial, action_str=None):
        super().__init__(base_trial, action_str)

    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(collection="weights"):
            t = self.base_trial.tensor(tname)
            if torch.isnan(t):
                return True
        return False


class GradientExplodingRule(Rule):
    def __init__(self, base_trial, action_str=None, threshold=1e6):
        super().__init__(base_trial, action_str)
        self.threshold = float(threshold)

    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(collection="gradients"):
            t = self.base_trial.tensor(tname)
            abs_mean = t.reduction_value(step, "mean", abs=True)
            if abs_mean > self.threshold:
                return True
        return False


class GradientVanishingRule(Rule):
    def __init__(
        self,
        base_trial,
        # action_str='{"name": "StopTraining", "training_job_prefix": ""}',
        action_str=None,
        threshold=1e-6,
    ):
        super().__init__(base_trial, action_str)
        self.threshold = float(threshold)

    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(collection="gradients"):
            t = self.base_trial.tensor(tname)
            # check Nan
            # if torch.isnan(
            #     t
            # ):  # NOT working, argument 'input' (position 1) must be Tensor, not Tensor
            #     return True
            abs_mean = t.reduction_value(step, "mean", abs=True)
            # check small gradients
            if abs_mean < self.threshold:
                print(f"abs_mean = {abs_mean}")
                return True
        return False


trial = create_trial(path=smdebug_dir)
print((trial.tensor_names()))

rule_obj = GradientVanishingRule(trial, threshold=1e-3)

try:
    invoke_rule(rule_obj, start_step=0, end_step=2840, raise_eval_cond=False)
except NoMoreData:
    print(
        "The training has ended and there is no more data to be analyzed. This is expected behavior."
    )

# values = trial.tensor("gradient/Net_conv1.bias").values()
# values_eval = np.array(list(values.items()))
# fig = plt.figure()
# plt.plot(values_eval[:, 1])
# fig.suptitle("Validation Accuracy", fontsize=20)
# plt.xlabel("Intervals of sampling", fontsize=18)
# plt.ylabel("Acuracy", fontsize=16)
# fig.savefig("temp.jpg")
