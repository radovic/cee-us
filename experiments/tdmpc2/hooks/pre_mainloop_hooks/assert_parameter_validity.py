import pickle

from mbrl import allogger
from smart_settings.param_classes import recursive_objectify

def assert_equals_parameters(_locals, _globals, **kwargs):
    
    assert _locals['params']['controller_params'].horizon == _locals['params']['forward_model_params']['model_params'].horizon, \
                "TDMPC2 requires the world model to have knowledge of the horizon parameter of the controller. Entry missing in .yaml or mismatch!"

    
