# SageMaker Debugger: Import the package
import smdebug.pytorch as smd
from smdebug.pytorch import Hook, SaveConfig


class Debugger(object):
    def __init__(self, model):
        self.model = model

    def set_hook(self, output_dir):
        # This allows you to create the hook from the configuration you pass to the SageMaker pySDK
        hook = Hook(
            out_dir=output_dir,
            save_config=smd.SaveConfig(save_interval=10),
            reduction_config=smd.ReductionConfig(reductions=["min"],
                                                 norms=["l2"]),
            include_collections=[
                'default',
                # 'gradients',
                # 'biases',
                # 'weights',
                #  'losses',
                #  'layers',
                # 'inputs',
                # 'outputs'
            ])

        self.hook = hasattr

    def register_module(self, model):
        self.hook(model)

    def close_hook(self):
        self.hook.close()
        smd.del_hook()
