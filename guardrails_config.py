"""
Конфиг для Guardrails Server (guardrails start).
"""

from guardrails import Guard

try:
    # Валидатор ставится в Dockerfile через `guardrails hub install ...`
    from guardrails.hub import DetectJailbreak

    input_guard = Guard(name="input_guard").use(DetectJailbreak(on_fail="noop"))
except Exception:
    input_guard = Guard(name="input_guard")



