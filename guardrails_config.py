"""
Конфиг для Guardrails Server (guardrails start).

В текущем docker-compose сервис Guardrails не собирается и не поднимается;
приложение при отсутствии GUARDRAILS_SERVER_URL не вызывает проверку входа.
"""

from guardrails import Guard
from typing import List

def set_guardrails(input_guard: List, output_guard: List):
    try:
        # Валидатор ставится в Dockerfile через `guardrails hub install ...`
        from guardrails.hub import DetectJailbreak

        input_guard = Guard(name="input_guard")
        input_guard = input_guard.use(DetectJailbreak(on_fail="noop"))
    except Exception as e:
        return e

    try:
        from guardrails.hub import ...

        output_guard = Guard(name="output_guard")
        output_guard = ...

    except Exception as e:
        return e
    return [input_guard, output_guard]

