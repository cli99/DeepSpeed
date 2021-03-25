"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

#########################################
# flops profiler
#########################################
# Flops profiler. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
FLOPS_PROFILER_FORMAT = '''
flops profiler should be enabled as:
"session_params": {
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    }
}
'''

FLOPS_PROFILER = "flops_profiler"

FLOPS_PROFILER_ENABLED = "enabled"
FLOPS_PROFILER_ENABLED_DEFAULT = False

FLOPS_PROFILER_PROFILE_STEP = "profile_step"
FLOPS_PROFILER_PROFILE_STEP_DEFAULT = 1

FLOPS_PROFILER_MODULE_DEPTH = "module_depth"
FLOPS_PROFILER_MODULE_DEPTH_DEFAULT = -1

FLOPS_PROFILER_TOP_MODULES = "top_modules"
FLOPS_PROFILER_TOP_MODULES_DEFAULT = 3

FLOPS_PROFILER_DETAILED = "detailed"
FLOPS_PROFILER_DETAILED_DEFAULT = True

#########################################
# XSP: across-stack profiler
#########################################
# XSP. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
XSP_FORMAT = '''
xsp should be enabled as:
"session_params": {
  "xsp": {
    "enalbe": [true|false],
    "level": 0,
    "max_event_duration": 0.5,
    "show_stack": [true|false],
    "start_step": 5,
    "end_step": 6,
    }
}
'''

XSP = "xsp"

XSP_ENABLED = "enabled"
XSP_ENABLED_DEFAULT = False

XSP_LEVEL = "level"
XSP_LEVEL_DEFAULT = 0

XSP_MAX_EVENT_DURATION = "max_event_duration"
XSP_MAX_EVENT_DURATION_DEFAULT = 0.5

XSP_SHOW_STACK = "show_stack"
XSP_SHOW_STACK_DEFAULT = False

XSP_START_STEP = "start_step"
XSP_START_STEP_DEFAULT = 5

XSP_END_STEP = "end_step"
XSP_END_STEP_DEFAULT = XSP_START_STEP_DEFAULT + 1
