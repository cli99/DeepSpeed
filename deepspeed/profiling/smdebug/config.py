# https://github.com/awslabs/sagemaker-debugger/blob/master/tests/pytorch/test_json_configs/test_hook_multi_collections.json

from deepspeed.runtime.config_utils import get_scalar_param
from deepspeed.profiling.constants import *

SMDEBUG_PROFILER_FORMAT = '''
smdebug should be enabled as:
{
  "session_params":{
    "smdebug":{
      "enable":[
        "true|false"
      ],
      "local_path":"/tmp/smdebug_output/",
      "export_tensorboard":true,
      "tensorboard_dir":"/tmp/tensorboard_dir",
      "hook_parameters":{
        "save_all":false,
        "reductions":"max, mean, variance",
        "save_steps":"0,1,2,3",
        "save_interval":10
      },
      "CollectionConfigurations":[
        {
          "CollectionName":"weights"
        },
        {
          "CollectionName":"biases"
        },
        {
          "CollectionName":"gradients"
        },
        {
          "CollectionName":"default"
        },
        {
          "CollectionName":"ReluActivation",
          "CollectionParameters":{
            "include_regex":"relu*",
            "save_steps":"4, 5, 6"
          }
        },
        {
          "CollectionName":"fc1",
          "CollectionParameters":{
            "include_regex":"fc1*",
            "save_steps":"7, 8, 9"
          }
        }
      ]
    }
  }
}
'''

SMDEBUG = "smdebug"

SMDEBUG_ENABLED = "enabled"
SMDEBUG_ENABLED_DEFAULT = False


class DeepSpeedSmdebugConfig(object):
    def __init__(self, param_dict):
        """
        docstring
        """
        super(DeepSpeedSmdebugConfig, self).__init__()

        self.enabled = None
        self.local_path = None
        self.export_tensorboard = None
        self.tensorboard_dir = None
        self.save_all = None

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

        hook_parameters_dict = smdebug_dict['hook_parameters']
        self.save_all = get_scalar_param(hook_parameters_dict,
                                         SMDEBUG_SAVE_ALL,
                                         SMDEBUG_SAVE_ALL)
