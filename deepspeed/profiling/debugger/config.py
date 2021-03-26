# https://github.com/awslabs/sagemaker-debugger/blob/master/tests/pytorch/test_json_configs/test_hook_multi_collections.json

from deepspeed.runtime.config_utils import get_scalar_param, get_list_param

DEBUGGER_FORMAT = '''
debugger should be enabled as:
{
  "session_params":{
    "debugger":{
      "enable": true,
      "output_dir": "/tmp/debugger_output/",
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

DEBUGGER = "debugger"

DEBUGGER_ENABLED = "enabled"
DEBUGGER_ENABLED_DEFAULT = False

DEBUGGER_OUTPUT_DIR = "output_dir"
DEBUGGER_OUTPUT_DIR_DEFAULT = None

DEBUGGER_EXPORT_TENSORBOARD = "enabled"
DEBUGGER_EXPORT_TENSORBOARD_DEFAULT = False

# DEBUGGER_TENSORBOARD_DIR = "export_tensorboard"
# DEBUGGER_TENSORBOARD_DIR_DEFAULT = "/tmp/tensorboard"

# logs weights, biases, gradients and inputs/ouputs of the model
DEBUGGER_SAVE_ALL = "save_all"
DEBUGGER_SAVE_ALL_DEFAULT = False

DEBUGGER_SAVE_STEPS = "save_steps"
DEBUGGER_SAVE_STEPS_DEFAULT = None

DEBUGGER_SAVE_INTERVAL = "save_interval"
DEBUGGER_SAVE_INTERVAL_DEFAULT = 10

DEBUGGER_COLLECTIONS = "collections"
DEBUGGER_COLLECTIONS_DEFAULT = None

DEBUGGER_REDUCTIONS = "reductions"
DEBUGGER_REDUCTIONS_DEFAULT = None

DEBUGGER_NORMS = "norms"
DEBUGGER_NORMS_DEFAULT = None

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

        if DEBUGGER in param_dict.keys():
            debugger_dict = param_dict[DEBUGGER]
        else:
            debugger_dict = {}

        self._initialize(debugger_dict)

    def _initialize(self, debugger_dict):
        """
        docstring
        """
        self.enabled = get_scalar_param(debugger_dict,
                                        DEBUGGER_ENABLED,
                                        DEBUGGER_ENABLED_DEFAULT)

        self.output_dir = get_scalar_param(debugger_dict,
                                           DEBUGGER_OUTPUT_DIR,
                                           DEBUGGER_OUTPUT_DIR_DEFAULT)

        self.export_tensorboard = get_scalar_param(debugger_dict,
                                                   DEBUGGER_EXPORT_TENSORBOARD,
                                                   DEBUGGER_EXPORT_TENSORBOARD_DEFAULT)

        # self.tensorboard_dir = get_scalar_param(debugger_dict,
        #                                         DEBUGGER_TENSORBOARD_DIR,
        #                                         DEBUGGER_TENSORBOARD_DIR_DEFAULT)

        hook_parameters_dict = debugger_dict.get('hook_parameters', None)
        # print(hook_parameters_dict)
        if hook_parameters_dict:
            self.save_all = get_scalar_param(hook_parameters_dict,
                                             DEBUGGER_SAVE_ALL,
                                             DEBUGGER_SAVE_ALL_DEFAULT)

            # https://github.com/awslabs/sagemaker-debugger/blob/master/debugger/core/save_config.py
            self.save_interval = get_scalar_param(hook_parameters_dict,
                                                  DEBUGGER_SAVE_INTERVAL,
                                                  DEBUGGER_SAVE_INTERVAL_DEFAULT)
            self.collections = parse_list(
                get_scalar_param(hook_parameters_dict,
                                 DEBUGGER_COLLECTIONS,
                                 DEBUGGER_COLLECTIONS_DEFAULT))

            # https://github.com/awslabs/sagemaker-debugger/blob/master/debugger/core/reduction_config.py
            self.reductions = parse_list(
                get_scalar_param(hook_parameters_dict,
                                 DEBUGGER_REDUCTIONS,
                                 DEBUGGER_REDUCTIONS_DEFAULT))

            self.norms = parse_list(
                get_scalar_param(hook_parameters_dict,
                                 DEBUGGER_NORMS,
                                 DEBUGGER_NORMS_DEFAULT))
