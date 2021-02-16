from smdebug.rules import Rule, invoke_rule
from smdebug.trials import create_trial
import numpy as np
import matplotlib.pyplot as plt

# https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html#exploding-tensor


class CustomGradientRule(Rule):
    def __init__(self, base_trial, threshold=10.0):
        super().__init__(base_trial)
        self.threshold = float(threshold)

    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(collection="gradients"):
            t = self.base_trial.tensor(tname)
            abs_mean = t.reduction_value(step, "mean", abs=True)
            if abs_mean > self.threshold:
                return True
        return False

        from smdebug.rules import Rule, invoke_rule


class CustomGradientRule(Rule):
    def __init__(self, base_trial, threshold=10.0):
        super().__init__(base_trial)
        self.threshold = float(threshold)

    def invoke_at_step(self, step):
        for tname in self.base_trial.tensor_names(collection="gradients"):
            t = self.base_trial.tensor(tname)
            abs_mean = t.reduction_value(step, "mean", abs=True)
            if abs_mean > self.threshold:
                return True
        return False


smdebug_dir = './output/mnist'

trial = create_trial(path=smdebug_dir)
rule_obj = CustomGradientRule(trial, threshold=0.0001)
invoke_rule(rule_obj, start_step=0, end_step=None)
