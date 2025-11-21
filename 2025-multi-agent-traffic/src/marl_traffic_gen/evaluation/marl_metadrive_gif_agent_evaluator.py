from agent_trainers.evaluators.base_agent_evaluator import AgentEvaluator
from agent_trainers.trainers import RllibAgentTrainer, Sb3AgentTrainer
from ray.rllib.algorithms import Algorithm
from stable_baselines3.common.base_class import BaseAlgorithm

from marl_traffic_gen.environments import RLlibMultiAgentMetaDrive


class MARLMetaDriveGifAgentEvaluator(AgentEvaluator):
    def evaluate_agent(
        self,
        agent_trainer: RllibAgentTrainer | Sb3AgentTrainer,
        environment: RLlibMultiAgentMetaDrive,
        policy: Algorithm | BaseAlgorithm,
    ) -> str:
        self.logger.info("Starting evaluation ...")

        env = environment
        for scenario_idx in range(
            env.unwrapped.env.start_index, env.unwrapped.env.start_index + env.unwrapped.env.num_scenarios
        ):
            cumulative_reward = {"agent0": 0.0, "agent1": 0.0}
            step: int = 0

            obs, _ = env.reset(seed=scenario_idx)
            while True:
                actions = {}
                for p, a_space, o in zip(policy, env.action_space, obs, strict=False):
                    action = agent_trainer.predict_action(policy[p], env.action_space[a_space], obs[o])
                    actions[a_space] = action

                obs, reward, terminated, truncated, _ = env.step(actions)
                cumulative_reward["agent0"] += reward["agent0"]
                cumulative_reward["agent1"] += reward["agent1"]
                step += 1

                done = {agent_id: terminated.get(agent_id, False) or truncated.get(agent_id, False) for agent_id in obs}
                done["__all__"] = any(done.values())
                if done["__all__"]:
                    break
            self.logger.info(
                "Scenario %d finished after %d steps with a cumulative_reward of agent0 %.3f and agent1 %.3f",
                scenario_idx,
                step,
                cumulative_reward["agent0"],
                cumulative_reward["agent1"],
            )
        return ""
