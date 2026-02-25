from __future__ import annotations



class InvalidPolicyError(Exception):
    def __init__(self, policy: str) -> None:
        self.policy = policy
        super().__init__(f"Invalid policy provided: {policy}")