# https://github.com/awslabs/sagemaker-debugger/blob/master/tests/pytorch/test_json_configs/test_hook_multi_collections.json

from deepspeed.runtime.config_utils import get_scalar_param, get_list_param
# from deepspeed.profiling.constants import *

SMDEBUG_PROFILER_FORMAT = '''
smdebug should be enabled as:
{
  "session_params":{
    "smdebug":{
      "enable": true,
      "output_dir": "/tmp/smdebug_output/",
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

SMDEBUG_OUTPUT_DIR = "output_dir"
SMDEBUG_OUTPUT_DIR_DEFAULT = None

SMDEBUG_EXPORT_TENSORBOARD = "enabled"
SMDEBUG_EXPORT_TENSORBOARD_DEFAULT = False

# SMDEBUG_TENSORBOARD_DIR = "export_tensorboard"
# SMDEBUG_TENSORBOARD_DIR_DEFAULT = "/tmp/tensorboard"

# logs weights, biases, gradients and inputs/ouputs of the model
SMDEBUG_SAVE_ALL = "save_all"
SMDEBUG_SAVE_ALL_DEFAULT = False

SMDEBUG_SAVE_STEPS = "save_steps"
SMDEBUG_SAVE_STEPS_DEFAULT = None

SMDEBUG_SAVE_INTERVAL = "save_interval"
SMDEBUG_SAVE_INTERVAL_DEFAULT = 10

SMDEBUG_COLLECTIONS = "collections"
SMDEBUG_COLLECTIONS_DEFAULT = None

SMDEBUG_REDUCTIONS = "reductions"
SMDEBUG_REDUCTIONS_DEFAULT = None

SMDEBUG_NORMS = "norms"
SMDEBUG_NORMS_DEFAULT = None

ALLOWED_REDUCTIONS = ["min", "max", "mean", "std", "variance", "sum", "prod"]
ALLOWED_NORMS = ["l1", "l2"]
ALLOWED_COLLECTIONS = [
    "weights",
    "gradients",
    "biases",
    "losses", # losses are logged in all cases
    # "default", # default does not output nothing
]


def parse_list(str):
    lst = str.replace(' ', '').split(",") if str else None
    return lst


class DeepSpeedDebuggerConfig(object):
    def __init__(self, param_dict):
        """
        docstring
        """
        super(DeepSpeedDebuggerConfig, self).__init__()

        self.enabled = None
        self.output_dir = None
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

        self.output_dir = get_scalar_param(smdebug_dict,
                                           SMDEBUG_OUTPUT_DIR,
                                           SMDEBUG_OUTPUT_DIR_DEFAULT)

        self.export_tensorboard = get_scalar_param(smdebug_dict,
                                                   SMDEBUG_EXPORT_TENSORBOARD,
                                                   SMDEBUG_EXPORT_TENSORBOARD_DEFAULT)

        # self.tensorboard_dir = get_scalar_param(smdebug_dict,
        #                                         SMDEBUG_TENSORBOARD_DIR,
        #                                         SMDEBUG_TENSORBOARD_DIR_DEFAULT)

        hook_parameters_dict = smdebug_dict.get('hook_parameters', None)
        # print(hook_parameters_dict)

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

        self.norms = parse_list(
            get_scalar_param(hook_parameters_dict,
                             SMDEBUG_NORMS,
                             SMDEBUG_NORMS_DEFAULT))
