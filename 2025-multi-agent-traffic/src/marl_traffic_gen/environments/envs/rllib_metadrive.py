from typing import Any

from metadrive.envs import MetaDriveEnv


class RLlibMetaDrive(MetaDriveEnv):
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict | Any] | tuple[Any, dict | Any]:
        return super().reset(seed=seed)
