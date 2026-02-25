from __future__ import annotations

from base.interfaces._evaluation_function import EvaluationFunction

class InvalidEvaluationFunctionError(Exception):
    def __init__(self, eval_func: EvaluationFunction) -> None:
        self.eval_func = eval_func
        super().__init__(f"Invalid evaluation function provided: {eval_func}")