from __future__ import annotations
import typing

class NotCallableError(Exception):
    def __init__(self, obj: typing.Any) -> None:
        self.obj = obj
        super().__init__(f"Object is not callable: {obj}. It must be a callable object.")