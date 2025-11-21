"""Register custom environments for scenario generation."""

from metadrive import MetaDriveEnv, MultiAgentMetaDrive

from marl_traffic_gen.environments.envs import RLlibMetaDrive, RLlibMultiAgentMetaDrive

import gymnasium

from ray.tune.registry import register_env


def register_envs() -> None:
    """Register all environment IDs with Gymnasium and RLlib."""

    # Gymnasium registrations
    gymnasium.register(id="MetaDriveEnv", entry_point=MetaDriveEnv)
    gymnasium.register(id="MultiAgentMetaDriveEnv", entry_point=MultiAgentMetaDrive)
    gymnasium.register(id="RLlibMultiAgentMetaDrive", entry_point=RLlibMultiAgentMetaDrive)

    # Ray Tune registrations (for RLlib)
    register_env("MetaDriveEnv", lambda config: MetaDriveEnv(config))
    register_env("MultiAgentMetaDriveEnv", lambda config: MultiAgentMetaDrive(config))
    register_env("RLlibMetaDriveEnv", lambda config: RLlibMetaDrive(config))
    register_env("RLlibMultiAgentMetaDrive", lambda config: RLlibMultiAgentMetaDrive(config))


__all__ = ["register_envs"]
