from typing import Any

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from marl_traffic_gen.environments.envs.metadrive_environment import CustomMultiAgentMetaDrive


class RLlibMultiAgentMetaDrive(MultiAgentEnv):
    """A wrapper to adapt the MetaDrive environment for use with RLlib's MultiAgentEnv interface."""

    def __init__(self, config: dict) -> None:
        super().__init__()

        # Initialize the MetaDrive environment
        self.env = CustomMultiAgentMetaDrive(config)
        # Set agent identifiers
        self.agents = self.possible_agents = ["agent0", "agent1"]
        # Set observation and action spaces (per-agent)
        self.observation_spaces = self.env.observation_space
        self.action_spaces = self.env.action_space
        # These are required by older RLlib MultiAgentEnv checkers
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict | Any] | tuple[Any, dict | Any]:
        return self.env.reset(seed=seed, options=options)

    def step(
        self, actions: dict
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        return self.env.step(actions)
