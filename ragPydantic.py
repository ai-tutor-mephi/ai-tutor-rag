from pydantic import BaseModel
from types import List

# Создание Pydantic классов
class ContentItem(BaseModel):
    fileId: str
    fileName: str
    text: str

class LoadRequest(BaseModel):
    content: List[ContentItem]
    dialogId: str

class DialogMessage(BaseModel):
    message: str
    role: str

class QueryRequest(BaseModel):
    dialogId: str
    dialogMessages: List[DialogMessage]
    question: str
