from pydantic import BaseModel, Field

class GraphGroundedAnswer(BaseModel):
    answer: str = Field(..., description="Синтезированный ответ на основе графового контекста")
    is_valid_output: bool = Field(..., description="True, если контекста достаточно; False, если информации не хватает")
    new_query: str = Field(..., description="Переформулированный запрос для повторного поиска контекста")
