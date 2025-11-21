from typing import Any

import numpy as np
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.constants import TerminationState
from metadrive.envs.marl_envs import MultiAgentMetaDrive
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import clip


class EgoVehicleManager(BaseManager):
    def reset(self) -> None:
        super().reset()
        # ✅ Spawn a separate ego vehicle with IDMPolicy (not controlled by RLlib)
        ego_vehicle = self.spawn_object(
            DefaultVehicle,
            vehicle_config={"use_special_color": True},
            position=(2, 0),
            heading=0,
        )
        self.add_policy(ego_vehicle.id, IDMPolicy, ego_vehicle, self.generate_seed())
        self.ego_vehicle = ego_vehicle

    def before_step(self) -> None:
        # Let the ego vehicle act using its own policy (IDM)
        for idx, vehicle in self.spawned_objects.items():
            policy = self.get_policy(idx)
            if policy is not None:
                vehicle.before_step(policy.act())

    def after_step(self, *args: str | int | float, **kwargs: str | int | float) -> None:
        for vehicle in self.spawned_objects.values():
            vehicle.after_step()


class CustomMultiAgentMetaDrive(MultiAgentMetaDrive):
    def setup_engine(self) -> None:
        # Keep default managers like VehicleAgentManager
        super().setup_engine()
        # Add our custom traffic manager (ego vehicle)
        self.engine.register_manager("ego_vehicle_manager", EgoVehicleManager())

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict | Any] | tuple[Any, dict | Any]:
        return super().reset(seed=seed)

    def reward_function(self, agent_id: str) -> tuple[float, dict]:
        reward = 0.0
        step_info = {}
        agent = self.agents[agent_id]
        ego = self.engine.ego_vehicle_manager.ego_vehicle

        # Road alignment and progress reward
        if agent.lane in agent.navigation.current_ref_lanes:
            current_lane = agent.lane
            positive_road = 1
        else:
            current_lane = agent.navigation.current_ref_lanes[0]
            current_road = agent.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1

        long_last, _ = current_lane.local_coordinates(agent.last_position)
        long_now, lateral_now = current_lane.local_coordinates(agent.position)

        lateral_factor = 1.0
        if self.config.get("use_lateral_reward", False):
            lane_width = agent.navigation.get_current_lane_width()
            lateral_factor = clip(1 - 2 * abs(lateral_now) / lane_width, 0.0, 1.0)

        reward += self.config.get("driving_reward", 1.0) * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config.get("speed_reward", 1.0) * (agent.speed_km_h / agent.max_speed_km_h) * positive_road

        # Negative events
        if agent.speed < 0.1:
            reward = -self.config.get("out_of_road_penalty", 5)
        if self._is_arrive_destination(agent):
            reward += self.config.get("success_reward", 10)
        elif self._is_out_of_road(agent):
            reward -= self.config.get("out_of_road_penalty", 5)
        elif agent.crash_vehicle:
            reward -= self.config.get("crash_vehicle_penalty", 5)
        elif agent.crash_object:
            reward -= self.config.get("crash_object_penalty", 5)
        elif agent.crash_sidewalk:
            reward -= self.config.get("crash_sidewalk_penalty", 2)

        # Special reward: agent hits the ego vehicle
        if ego.crash_vehicle and agent.crash_vehicle:
            reward += self.config.get("success_reward", 10)
            step_info["crash_with"] = "ego_vehicle"

        step_info["step_reward"] = reward
        step_info["route_completion"] = agent.navigation.route_completion
        return reward, step_info

    def done_function(self, agent_id: str) -> tuple[bool, dict]:
        agent = self.agents[agent_id]
        ego = self.engine.ego_vehicle_manager.ego_vehicle
        ego_pos = np.array(ego.position[:2])
        agent_pos = np.array(agent.position[:2])
        distance = np.linalg.norm(ego_pos - agent_pos)

        max_step = (
            self.config.get("horizon", None) is not None and self.episode_lengths[agent_id] >= self.config["horizon"]
        )

        done_info = {
            TerminationState.CRASH_VEHICLE: agent.crash_vehicle,
            TerminationState.CRASH_OBJECT: agent.crash_object,
            TerminationState.CRASH_BUILDING: agent.crash_building,
            TerminationState.CRASH_HUMAN: agent.crash_human,
            TerminationState.CRASH_SIDEWALK: agent.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(agent) or agent.navigation.route_completion < -0.1,
            TerminationState.SUCCESS: distance < getattr(self, "MIN_DISTANCE", 2.0),
            TerminationState.MAX_STEP: max_step,
        }

        done_info[TerminationState.CRASH] = any(
            [
                done_info[TerminationState.CRASH_VEHICLE],
                done_info[TerminationState.CRASH_OBJECT],
                done_info[TerminationState.CRASH_BUILDING],
                done_info[TerminationState.CRASH_SIDEWALK],
                done_info[TerminationState.CRASH_HUMAN],
            ]
        )

        done = any(
            [
                done_info[TerminationState.CRASH],
                done_info[TerminationState.OUT_OF_ROAD],
                done_info[TerminationState.SUCCESS],
                done_info[TerminationState.MAX_STEP],
                ego.crash_vehicle,
            ]
        )

        if done_info[TerminationState.MAX_STEP] and self.config.get("truncate_as_terminate", False):
            done = True

        return done, done_info
