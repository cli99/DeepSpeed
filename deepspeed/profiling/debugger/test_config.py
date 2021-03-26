import os
import torch
import pytest
import json
import argparse

# import deepspeed
# from deepspeed.runtime.config import DeepSpeedConfig

import deepspeed.profiling.debugger as db

tmpdir = os.path.dirname(os.path.realpath(__file__))

print("tmpdir is ", tmpdir)


def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


def test_temp_config_json():
    config_dict = {
        "debugger": {
            "enable": True,
            "local_path": "dfs",
            "export_tensorboard": True,
            "hook_parameters": {
                "save_all": True,
                "collections": "weights, gradients, biases, inputs, outputs",
                "reductions": "max, mean, variance",
                "save_steps": "0,1,2,3",
                "save_interval": 10
            },
        }
    }
    config_path = create_config_from_dict(tmpdir, config_dict)
    config_json = json.load(open(config_path, 'r'))
    config = db.DeepSpeedDebuggerConfig(config_json)
    print(config)

    assert config.collections == "weights, gradients, biases, inputs, outputs"

    assert False
