# DDPG
```
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:131: UserWarning: WARN: The obs returned by the `reset()` method was expecting a numpy array, actual type: <class 'torch.Tensor'>
  logger.warn(
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/spaces/box.py:240: UserWarning: WARN: Casting input x to numpy array.
  gym.logger.warn("Casting input x to numpy array.")
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:159: UserWarning: WARN: The obs returned by the `reset()` method is not within the observation space.
  logger.warn(f"{pre} is not within the observation space.")
  0%|                                                                                                                                           | 0/14000 [00:00<?, ?it/s]/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:228: UserWarning: WARN: Expects `terminated` signal to be a boolean, actual type: <class 'torch.Tensor'>
  logger.warn(
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:232: UserWarning: WARN: Expects `truncated` signal to be a boolean, actual type: <class 'torch.Tensor'>
  logger.warn(
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:131: UserWarning: WARN: The obs returned by the `step()` method was expecting a numpy array, actual type: <class 'torch.Tensor'>
  logger.warn(
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:159: UserWarning: WARN: The obs returned by the `step()` method is not within the observation space.
  logger.warn(f"{pre} is not within the observation space.")
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:246: UserWarning: WARN: The reward returned by `step()` must be a float, int, np.integer or np.floating, actual type: <class 'torch.Tensor'>
  logger.warn(
 29%|████████████████████████████████████▋                                                                                           | 4011/14000 [01:49<04:31, 36.79it/s]/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:327: UserWarning: WARN: Expects all render modes to be strings, actual types: [<class 'NoneType'>, <class 'str'>, <class 'str'>]
  logger.warn(
2025-05-14 11:16:19 [125,052ms] [Warning] [rtx.postprocessing.plugin] DLSS increasing input dimensions: Render resolution of (320, 240) is below minimal input resolution of 300.
```

# PPO

```
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:131: UserWarning: WARN: The obs returned by the `reset()` method was expecting a numpy array, actual type: <class 'torch.Tensor'>
  logger.warn(
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/spaces/box.py:240: UserWarning: WARN: Casting input x to numpy array.
  gym.logger.warn("Casting input x to numpy array.")
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:159: UserWarning: WARN: The obs returned by the `reset()` method is not within the observation space.
  logger.warn(f"{pre} is not within the observation space.")
  0%|                                                                                                                                           | 0/14000 [00:00<?, ?it/s]/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:228: UserWarning: WARN: Expects `terminated` signal to be a boolean, actual type: <class 'torch.Tensor'>
  logger.warn(
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:232: UserWarning: WARN: Expects `truncated` signal to be a boolean, actual type: <class 'torch.Tensor'>
  logger.warn(
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:131: UserWarning: WARN: The obs returned by the `step()` method was expecting a numpy array, actual type: <class 'torch.Tensor'>
  logger.warn(
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:159: UserWarning: WARN: The obs returned by the `step()` method is not within the observation space.
  logger.warn(f"{pre} is not within the observation space.")
/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:246: UserWarning: WARN: The reward returned by `step()` must be a float, int, np.integer or np.floating, actual type: <class 'torch.Tensor'>
  logger.warn(
 43%|███████████████████████████████████████████████████████▏                                                                        | 6033/14000 [02:02<02:26, 54.54it/s]/home/user/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/utils/passive_env_checker.py:327: UserWarning: WARN: Expects all render modes to be strings, actual types: [<class 'NoneType'>, <class 'str'>, <class 'str'>]
  logger.warn(
 43%|███████████████████████████████████████████████████████▏                                                                        | 6039/14000 [02:03<05:57, 22.26it/s]2025-05-14 11:24:56 [142,897ms] [Warning] [rtx.postprocessing.plugin] DLSS increasing input dimensions: Render resolution of (320, 240) is below minimal input resolution of 300.

```