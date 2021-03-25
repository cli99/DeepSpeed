"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigObject
from deepspeed.profiling.constants import *


class DeepSpeedFlopsProfilerConfig(DeepSpeedConfigObject):
    def __init__(self, param_dict):
        """
        docstring
        """
        super(DeepSpeedFlopsProfilerConfig, self).__init__()

        self.enabled = None
        self.profile_step = None
        self.module_depth = None
        self.top_modules = None

        if FLOPS_PROFILER in param_dict.keys():
            flops_profiler_dict = param_dict[FLOPS_PROFILER]
        else:
            flops_profiler_dict = {}

        self._initialize(flops_profiler_dict)

    def _initialize(self, flops_profiler_dict):
        """
        docstring
        """
        self.enabled = get_scalar_param(flops_profiler_dict,
                                        FLOPS_PROFILER_ENABLED,
                                        FLOPS_PROFILER_ENABLED_DEFAULT)

        self.profile_step = get_scalar_param(flops_profiler_dict,
                                             FLOPS_PROFILER_PROFILE_STEP,
                                             FLOPS_PROFILER_PROFILE_STEP_DEFAULT)

        self.module_depth = get_scalar_param(flops_profiler_dict,
                                             FLOPS_PROFILER_MODULE_DEPTH,
                                             FLOPS_PROFILER_MODULE_DEPTH_DEFAULT)

        self.top_modules = get_scalar_param(flops_profiler_dict,
                                            FLOPS_PROFILER_TOP_MODULES,
                                            FLOPS_PROFILER_TOP_MODULES_DEFAULT)

        self.detailed = get_scalar_param(flops_profiler_dict,
                                         FLOPS_PROFILER_DETAILED,
                                         FLOPS_PROFILER_DETAILED_DEFAULT)


class DeepSpeedXSPConfig(object):
    def __init__(self, param_dict):
        super(DeepSpeedXSPConfig, self).__init__()

        self.enabled = None
        self.level = None
        self.show_stack = None
        self.max_event_duration = None

        if XSP in param_dict.keys():
            xsp_dict = param_dict[XSP]
        else:
            xsp_dict = {}

        self._initialize(xsp_dict)

    def _initialize(self, xsp_dict):
        self.enabled = get_scalar_param(xsp_dict, XSP_ENABLED, XSP_ENABLED_DEFAULT)
        self.level = get_scalar_param(xsp_dict, XSP_LEVEL, XSP_LEVEL_DEFAULT)
        self.max_event_duration = get_scalar_param(xsp_dict,
                                                   XSP_MAX_EVENT_DURATION,
                                                   XSP_MAX_EVENT_DURATION_DEFAULT)
        self.show_stack = get_scalar_param(xsp_dict,
                                           XSP_SHOW_STACK,
                                           XSP_SHOW_STACK_DEFAULT)
        self.start_step = get_scalar_param(xsp_dict,
                                           XSP_START_STEP,
                                           XSP_START_STEP_DEFAULT)

        self.end_step = get_scalar_param(xsp_dict, XSP_END_STEP, XSP_END_STEP_DEFAULT)
