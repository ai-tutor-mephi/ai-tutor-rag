"""
Конфиг для Guardrails
"""
from guardrails import Guard
from guardrails.hub import DetectJailbreak

input_guard = Guard().use(DetectJailbreak(on_fail="noop"))



