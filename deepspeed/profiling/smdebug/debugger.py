import atexit

# SageMaker Debugger: Import the package
import smdebug.pytorch as smd
from smdebug.pytorch import Hook, SaveConfig


class Debugger(object):
    def __init__(self):
        atexit.register(self.close)

    def register_module(self,
                        model,
                        output_dir="./debugger",
                        export_tensorboard=False,
                        save_all=True,
                        collections=None,
                        reductions=None,
                        norms=None,
                        save_interval=None):
        # Create the hook from the configuration you pass to the SageMaker pySDK
        self.hook = Hook(out_dir=output_dir,
                         save_all=save_all,
                         save_config=smd.SaveConfig(save_interval=save_interval),
                         reduction_config=smd.ReductionConfig(reductions=reductions,
                                                              norms=norms),
                         include_collections=collections,
                         export_tensorboard=export_tensorboard)

        self.hook.register_module(model)

    def close(self):
        self.hook.close()
        smd.del_hook()
