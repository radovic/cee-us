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
        rel_time = np.mean( np.sum(moved_objects_indices[i,:], axis=0) / factor_for_relative_scaling )
        metrics[obj_name + '_rel_time'] = rel_time
        logger.log(rel_time, key=obj_name + '_rel_time')
    
    moved_at_least_one = np.sum(moved_objects_indices, axis=0) >= 1
    rel_time = np.mean(moved_at_least_one / factor_for_relative_scaling)
    metrics['any_object_rel_time'] = rel_time
    logger.log(rel_time, key='any_object_rel_time')







