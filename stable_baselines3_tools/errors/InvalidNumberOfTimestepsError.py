from __future__ import annotations


class InvalidNumberOfTimestepsError(Exception):
    def __init__(self, timesteps: int) -> None:
        self.timesteps = timesteps
        super().__init__(f"Invalid number of timesteps provided: {timesteps}")