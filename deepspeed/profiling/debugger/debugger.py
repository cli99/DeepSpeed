import os
import atexit
import datetime

# SageMaker Debugger: Import the package
import smdebug.pytorch as smd
from smdebug.pytorch import Hook, SaveConfig

DEFAULT_OUTPUT_DIR = "/tmp/dsdebugger"


class Debugger(object):
    def __init__(self):
        atexit.register(self.close)

    def register_module(self,
                        model,
                        output_dir=DEFAULT_OUTPUT_DIR,
                        export_tensorboard=False,
                        save_all=True,
                        collections=None,
                        reductions=None,
                        norms=None,
                        save_interval=None):

        # if output_dir is not specified, output to the default location using the timestamp
        output_dir = os.path.join(output_dir,
                                  datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        # Create the hook from the configuration you pass to the SageMaker pySDK
        self.hook = Hook(
            out_dir=output_dir,
            save_all=save_all,
            save_config=smd.SaveConfig(save_interval=save_interval),
            #  reduction_config=smd.ReductionConfig(reductions=reductions,norms=norms),
            include_collections=collections,
            export_tensorboard=export_tensorboard)

        self.hook.register_module(model)

    def close(self):
        if self.hook:
            self.hook.close()
            smd.del_hook()
        else:
            print("there is no hook to close!")
