import importlib
import sys
from pathlib import Path

import hydra
from agent_trainers.environments import get_env
from agent_trainers.trainers import RllibAgentTrainer
from agent_trainers.utils.seeding import seed_everything
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from ray.rllib.policy.policy import PolicySpec
from rich.console import Console

from marl_traffic_gen.environments import register_envs
from marl_traffic_gen.evaluation import MARLMetaDriveGifAgentEvaluator

importlib.import_module("agent_trainers.hydra_custom_plugins")
importlib.import_module("agent_trainers.environments")

main_dir = Path(__file__).parent.resolve()
sys.path.append(main_dir.as_posix())

register_envs()


@hydra.main(
    config_path=(main_dir / "configs/").as_posix(),
    config_name="root",
    version_base="1.3",
)
def main(config: DictConfig) -> None:
    seed_everything(seed=config.seed)

    # Update multi_agent configuration of agent_kwargs
    env = get_env(**OmegaConf.to_container(config.environment, resolve=True), seed=config.seed).unwrapped
    agent_kwargs = OmegaConf.to_container(config.agent.agent_kwargs, resolve=True)
    agent_kwargs["multi_agent"]["policies"] = {
        "p0": PolicySpec(
            policy_class=None,
            observation_space=env.observation_spaces["agent0"],
            action_space=env.action_spaces["agent0"],
        ),
        "p1": PolicySpec(
            policy_class=None,
            observation_space=env.observation_spaces["agent1"],
            action_space=env.action_spaces["agent1"],
        ),
    }
    agent_kwargs["multi_agent"]["policy_mapping_fn"] = (
        lambda agent_id, episode, **kwargs: "p0" if agent_id == "agent0" else "p1"
    )
    agent_kwargs["multi_agent"]["policies_to_train"] = ["p0", "p1"]

    # Get the Reinforcement Learning Agent
    agent_trainer = RllibAgentTrainer(
        agent_class=config.agent.agent_class,
        agent_kwargs=agent_kwargs,
        agent_name=config.agent_name,
        environment_id=config.environment.environment_id,
        environment_kwargs=OmegaConf.to_container(
            config.environment.environment_kwargs.env_kwargs.config, resolve=True
        ),
        time_stamp=HydraConfig.get().runtime.output_dir,
        seed=config.seed,
        torch_device_name=config.torch_device_name,
    )

    if config.learn:
        # Train the Reinforcement Learning Agent
        agent_trainer.learn(
            **OmegaConf.to_container(config.learning, resolve=True),
        )

    if config.test:
        # Evaluate the Reinforcement Learning Agent
        testing_environment = get_env(
            **OmegaConf.to_container(config.testing.environment, resolve=True),
            seed=config.seed,
        )
        policy = agent_trainer.load_policy(
            policy_path=config.testing.policy.policy_path,
        )
        evaluator = MARLMetaDriveGifAgentEvaluator()
        evaluator.evaluate_agent(
            agent_trainer=agent_trainer,
            environment=testing_environment,
            policy=policy,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[yellow]Interrupted from user keyboard.")
    except Exception:
        Console(quiet=False).print_exception()
    finally:
        sys.exit()
