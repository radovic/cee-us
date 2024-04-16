import os
import numpy as np

def interaction_tracker_hook(_locals, _globals, **kwargs):

    logger = _locals["rollout_man"].logger
    metrics = _locals["metrics"]
    env = _locals["env"]
    latest_rollouts = _locals["buffer"]["rollouts"].latest_rollouts

    obs_state_dict = env.get_object_centric_obs(latest_rollouts["observations"])
    obs_state_dict_next = env.get_object_centric_obs(latest_rollouts["next_observations"])

    # successes = latest_rollouts["successes"].reshape(len(latest_rollouts), -1)
    # rollout_successes = np.clip(np.sum(successes, axis=-1), 0, 1)
    # final_successes = np.clip(successes[:, -1], 0, 1)
    # for idx in range(len(rollout_successes)): logger.log(rollout_successes[idx], key='success')
    # for idx in range(len(final_successes)): logger.log(final_successes[idx], key='final_success')
    
    objects_delta = obs_state_dict_next['objects_dyn'] - obs_state_dict['objects_dyn']

    moved_objects_indices = np.any(np.abs(objects_delta[...,:3])>1e-3, axis=-1) # num_obj x timesteps

    factor_for_relative_scaling = latest_rollouts["observations"].shape[0]

    for i, obj_name in enumerate(env.env_body_names):
        rel_time = np.sum(moved_objects_indices[i,:]) / factor_for_relative_scaling
        metrics[obj_name + '_rel_time'] = rel_time
        logger.log(rel_time, key=obj_name + '_rel_time')

def cube_off_table_tracker_hook(_locals, _globals, **kwargs):
    """
        Only applicable to franka_cube_move.py environment.
    """
    
    logger = _locals["logger"]
    obs = _locals['next_ob']
    env = _locals["env"]
    object_static = env.get_object_centric_obs(obs)['objects_static'][0,0]
    fell_off = object_static[2] < 0.5 # assume that the gripper may potentially reach lower than the table with cube in hand.
    logger.log(fell_off, key=env.env_body_names[0]+'_fell_off_table')



