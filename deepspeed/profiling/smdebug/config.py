# https://github.com/awslabs/sagemaker-debugger/blob/master/tests/pytorch/test_json_configs/test_hook_multi_collections.json

from deepspeed.runtime.config_utils import get_scalar_param, get_list_param
# from deepspeed.profiling.constants import *

SMDEBUG_PROFILER_FORMAT = '''
smdebug should be enabled as:
{
  "session_params":{
    "smdebug":{
      "enable": true,
      "local_path": "/tmp/smdebug_output/",
      "export_tensorboard":true,
      "tensorboard_dir": "/tmp/tensorboard_dir",
      "hook_parameters":{
        "save_all": false,
        "collections": "weights, gradients, biases, inputs, outputs",
        "reductions": "max, mean, variance",
        "save_steps": "0,1,2,3",
        "save_interval": 10
      },
  }
}
'''

SMDEBUG = "smdebug"

SMDEBUG_ENABLED = "enabled"
SMDEBUG_ENABLED_DEFAULT = False

SMDEBUG_LOCAL_PATH = "local_path"
SMDEBUG_LOCAL_PATH_DEFAULT = "/tmp/smdebug"

SMDEBUG_EXPORT_TENSORBOARD = "enabled"
SMDEBUG_EXPORT_TENSORBOARD_DEFAULT = False

SMDEBUG_TENSORBOARD_DIR = "export_tensorboard"
SMDEBUG_TENSORBOARD_DIR_DEFAULT = "/tmp/tensorboard"

SMDEBUG_SAVE_ALL = "save_all"
SMDEBUG_SAVE_ALL_DEFAULT = False

SMDEBUG_SAVE_STEPS = "save_steps"
SMDEBUG_SAVE_STEPS_DEFAULT = None

SMDEBUG_SAVE_INTERVALS = "save_intervals"
SMDEBUG_SAVE_INTERVALS_DEFAULT = 100

SMDEBUG_COLLECTIONS = "collections"
SMDEBUG_COLLECTIONS_DEFAULT = None

SMDEBUG_REDUCTIONS = "reductions"
SMDEBUG_REDUCTIONS_DEFAULT = None


def parse_list(str):
    return str.replace(' ', '').split(",")


class DeepSpeedDebuggerConfig(object):
    def __init__(self, param_dict):
        """
        docstring
        """
        super(DeepSpeedDebuggerConfig, self).__init__()

        self.enabled = None
        self.local_path = None
        self.export_tensorboard = None
        self.tensorboard_dir = None
        self.save_all = None
        self.save_interval = None
        self.collections = None
        self.reductions = None

        if SMDEBUG in param_dict.keys():
            smdebug_dict = param_dict[SMDEBUG]
        else:
            smdebug_dict = {}

        self._initialize(smdebug_dict)

    def _initialize(self, smdebug_dict):
        """
        docstring
        """
        self.enabled = get_scalar_param(smdebug_dict,
                                        SMDEBUG_ENABLED,
                                        SMDEBUG_ENABLED_DEFAULT)

        self.local_path = get_scalar_param(smdebug_dict,
                                           SMDEBUG_LOCAL_PATH,
                                           SMDEBUG_LOCAL_PATH_DEFAULT)

        self.export_tensorboard = get_scalar_param(smdebug_dict,
                                                   SMDEBUG_EXPORT_TENSORBOARD,
                                                   SMDEBUG_EXPORT_TENSORBOARD_DEFAULT)

        self.tensorboard_dir = get_scalar_param(smdebug_dict,
                                                SMDEBUG_TENSORBOARD_DIR,
                                                SMDEBUG_TENSORBOARD_DIR_DEFAULT)

        hook_parameters_dict = smdebug_dict.get('hook_parameters', None)
        print(hook_parameters_dict)

        self.save_all = get_scalar_param(hook_parameters_dict,
                                         SMDEBUG_SAVE_ALL,
                                         SMDEBUG_SAVE_ALL_DEFAULT)

        # https://github.com/awslabs/sagemaker-debugger/blob/master/smdebug/core/save_config.py
        self.save_interval = get_scalar_param(hook_parameters_dict,
                                              SMDEBUG_SAVE_INTERVAL,
                                              SMDEBUG_SAVE_INTERVAL_DEFAULT)
        self.collections = parse_list(
            get_scalar_param(hook_parameters_dict,
                             SMDEBUG_COLLECTIONS,
                             SMDEBUG_COLLECTIONS_DEFAULT))

        # https://github.com/awslabs/sagemaker-debugger/blob/master/smdebug/core/reduction_config.py
        self.reductions = parse_list(
            get_scalar_param(hook_parameters_dict,
                             SMDEBUG_REDUCTIONS,
                             SMDEBUG_REDUCTIONS_DEFAULT))
        print(self)
