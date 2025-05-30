from typing import Any, Mapping, Optional, Tuple, Union

import copy
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model


# fmt: off
# [start-config-dict-torch]
DDPG_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "exploration": {
        "noise": None,              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1e-3,        # final scale for the noise
        "timesteps": None,          # timesteps for the noise decay
    },

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class DDPG(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Deep Deterministic Policy Gradient (DDPG)

        https://arxiv.org/abs/1509.02971

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(DDPG_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        # self.target_policy = self.models.get("target_policy", None)
        self.critic = self.models.get("critic", None)
        self.target_critic = self.models.get("target_critic", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        # self.checkpoint_modules["target_policy"] = self.target_policy
        self.checkpoint_modules["critic"] = self.critic
        self.checkpoint_modules["target_critic"] = self.target_critic

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic is not None:
                self.critic.broadcast_parameters()

        # if self.target_policy is not None and self.target_critic is not None:
        #     # freeze target networks with respect to optimizers (update via .update_parameters())
        #     self.target_policy.freeze_parameters(True)
        #     self.target_critic.freeze_parameters(True)

        #     # update target networks (hard update)
        #     self.target_policy.update_parameters(self.policy, polyak=1)
        #     self.target_critic.update_parameters(self.critic, polyak=1)
        
        ####################################################################  I've reverted this part
        if self.target_critic is not None:
            self.target_critic.freeze_parameters(True)
            self.target_critic.update_parameters(self.critic, polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        # self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"].get("initial_scale", 1.0)
        self._exploration_final_scale = self.cfg["exploration"].get("final_scale", 1e-3)
        self._exploration_timesteps = self.cfg["exploration"].get("timesteps", None)

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic is not None:
            #self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self._critic_learning_rate)
            if self._learning_rate_scheduler is not None:
                # self.policy_scheduler = self._learning_rate_scheduler(
                #     self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                # )
                self.critic_scheduler = self._learning_rate_scheduler(
                    self.critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            #self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory   BUFFER
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            # self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)      # not
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.int)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)  # maybe to remove

            self._tensors_names = ["states",  "rewards", "next_states", "terminated", "truncated"]
                                             #"actions",
        # clip noise bounds
        # if self.action_space is not None:
        #     self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
        #     self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample DETERMINISTIC actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, _, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
        # HERE THE ACTIONS ARE THE ONES THAT THE ROBOT HAS TO DO, COMING FROM THE ALREADY TRAINED POLICY!!!!
        
        # add exloration noise
        if self._exploration_noise is not None:
            # sample noises
            noises = self._exploration_noise.sample(actions.shape)

            # define exploration timesteps
            scale = self._exploration_final_scale
            if self._exploration_timesteps is None:
                self._exploration_timesteps = timesteps

            # apply exploration noise
             # if timestep <= self._exploration_timesteps:
                # NOISE SCALING 
                # scale = (1 - timestep / self._exploration_timesteps) * (
                #     self._exploration_initial_scale - self._exploration_final_scale
                # ) + self._exploration_final_scale
                # noises.mul_(scale)

                # modify actions
            
            #################################################################### Commented out but i am not sure!
            # actions.add_(noises)
            # actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

            # record noises
            self.track_data("Exploration / Exploration noise (max)", torch.max(noises).item())
            self.track_data("Exploration / Exploration noise (min)", torch.min(noises).item())
            self.track_data("Exploration / Exploration noise (mean)", torch.mean(noises).item())

            # else:
            #     # record noises
            #     self.track_data("Exploration / Exploration noise (max)", 0)
            #     self.track_data("Exploration / Exploration noise (min)", 0)
            #     self.track_data("Exploration / Exploration noise (mean)", 0)

        return actions, None, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor, # maybe to remove
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )        # !
        # actions (after states)

        if self.memory is not None:
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                # actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,   #
                truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    # actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,  #
                    truncated=truncated,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            # sample a (mini) batch from memory
            (
                sampled_states,
                # sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated, #
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

                # compute target values
                with torch.no_grad():
                    # next_actions, _, _ = self.target_policy.act({"states": sampled_next_states}, role="target_policy")

                    target_q_values, _, _ = self.target_critic.act(            # was target_critics
                        {"states": sampled_next_states}, role="critic"   # , "taken_actions": next_actions
                    )
                    reach_set = sampled_rewards%2
                    avoid_set = sampled_rewards//2
                    target_values = (
                        reach_set * (not avoid_set) * 1 + ((not reach_set) * (not avoid_set)) * target_q_values #+ avoid_set*0 
                        # PASS THEM AS BOOLEAN !!
                        # sampled_rewards * target_q_values
                        # * (sampled_terminated | sampled_truncated).logical_not()
                    )
                    # ABOVE TO DO
                        # target_value(x) = 0 if x in avoid_set,
                        # target_value(x) = 1 if x in reach_set,
                        # target_value(x) = target_critic(x’) otherwise


                # compute critic loss
                critic_values, _, _ = self.critic.act(
                    {"states": sampled_states}, role="critic"
                )

                critic_loss = F.mse_loss(critic_values, target_values)

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                self.critic.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self._grad_norm_clip)

            self.scaler.step(self.critic_optimizer)

            # with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            #     # compute policy (actor) loss
            #     actions, _, _ = self.policy.act({"states": sampled_states}, role="policy")
            #     critic_values, _, _ = self.critic.act(
            #         {"states": sampled_states, "taken_actions": actions}, role="critic"
            #     )

            #    policy_loss = -critic_values.mean()

            # # optimization step (policy)
            # self.policy_optimizer.zero_grad()
            # self.scaler.scale(policy_loss).backward()

            # if config.torch.is_distributed:
            #     self.policy.reduce_parameters()

            # if self._grad_norm_clip > 0:
            #     self.scaler.unscale_(self.policy_optimizer)
            #     nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            # self.scaler.step(self.policy_optimizer)

            self.scaler.update()  # called once, after optimizers have been stepped

            # update target networks
            # self.target_policy.update_parameters(self.policy, polyak=self._polyak)
            self.target_critic.update_parameters(self.critic, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                # self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            # self.track_data("Loss / Policy loss", policy_loss.item())
            self.track_data("Loss / Critic loss", critic_loss.item())

            self.track_data("Q-network / Q1 (max)",  torch.max(critic_values).item())
            self.track_data("Q-network / Q1 (min)",  torch.min(critic_values).item())
            self.track_data("Q-network / Q1 (mean)", torch.mean(critic_values).item())

            self.track_data("Target / Target (max)",  torch.max(target_values).item())
            self.track_data("Target / Target (min)",  torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self._learning_rate_scheduler:
                # self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])