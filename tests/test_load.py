"""
Тесты для сервиса загрузки документов.

Проверяют корректность работы LoadService:
- Разбиение текста на чанки
- Векторизация чанков
- Загрузка в Qdrant
- Создание графа в Neo4j
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.load_service import LoadService
from src.Handling.Chunker import Chunker
from src.Handling.Embedder import Embedder
from src.Databases.QInteracter import QInteracter
from src.Databases.NeoInteracter import NeoInteracter


@pytest.fixture
def mock_chunker():
    chunker = MagicMock(spec=Chunker)
    chunker.make_chunks_from_text = MagicMock(return_value=[
        "Это первый чанк текста.",
        "Это второй чанк текста.",
        "Это третий чанк текста."
    ])
    return chunker


@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=Embedder)

    embedder.embed = MagicMock(return_value=[0.1] * 1024)
    return embedder


@pytest.fixture
def mock_qdrant():
    qdrant = MagicMock(spec=QInteracter)
    qdrant.load_in_qdrant = AsyncMock(return_value=None)
    return qdrant


@pytest.fixture
def mock_neo():
    neo = MagicMock(spec=NeoInteracter)
    neo.create_graph = AsyncMock(return_value=None)
    return neo


@pytest.fixture
def load_service(mock_chunker, mock_embedder, mock_qdrant, mock_neo):
    return LoadService(
        chunker=mock_chunker,
        embedder=mock_embedder,
        qdrant=mock_qdrant,
        neo=mock_neo
    )


class TestLoadService:
    """Тесты для LoadService."""
    
    @pytest.mark.asyncio
    async def test_process_file_creates_chunks(self, load_service, mock_chunker):
        """Тест: процесс загрузки создает чанки из текста."""
        file_id = str(uuid.uuid4())
        file_name = "test_document.txt"
        text = "Это тестовый документ. Он содержит несколько предложений для разбиения на чанки."
        dialog_id = "test_dialog_123"
        
        await load_service.process_file(file_id, file_name, text, dialog_id)
        
        # Проверяем, что чанкер был вызван
        mock_chunker.make_chunks_from_text.assert_called_once_with(text)
    
    @pytest.mark.asyncio
    async def test_process_file_vectorizes_chunks(self, load_service, mock_embedder):
        """Тест: процесс загрузки векторизует все чанки."""
        file_id = str(uuid.uuid4())
        file_name = "test_document.txt"
        text = "Тестовый текст для векторизации."
        dialog_id = "test_dialog_123"
        
        await load_service.process_file(file_id, file_name, text, dialog_id)
        
        # Проверяем, что embedder был вызван для каждого чанка
        # (3 чанка из mock_chunker)
        assert mock_embedder.embed.call_count == 3
    
    @pytest.mark.asyncio
    async def test_process_file_loads_to_qdrant(self, load_service, mock_qdrant):
        """Тест: процесс загрузки сохраняет чанки в Qdrant."""
        file_id = str(uuid.uuid4())
        file_name = "test_document.txt"
        text = "Тестовый текст для загрузки в Qdrant."
        dialog_id = "test_dialog_123"
        
        await load_service.process_file(file_id, file_name, text, dialog_id)
        
        # Проверяем, что qdrant.load_in_qdrant был вызван
        mock_qdrant.load_in_qdrant.assert_called_once()
        
        # Проверяем, что в Qdrant были переданы чанки с правильными метаданными
        call_args = mock_qdrant.load_in_qdrant.call_args[0][0]
        assert len(call_args) == 3
        assert all("text" in chunk for chunk in call_args)
        assert all("dense_vector" in chunk for chunk in call_args)
        assert all("file_id" in chunk for chunk in call_args)
        assert all("file_name" in chunk for chunk in call_args)
        assert all("dialog_id" in chunk for chunk in call_args)
        assert all("chunk_id" in chunk for chunk in call_args)
    
    @pytest.mark.asyncio
    async def test_process_file_creates_graph_in_neo4j(self, load_service, mock_neo):
        """Тест: процесс загрузки создает граф в Neo4j."""
        file_id = str(uuid.uuid4())
        file_name = "test_document.txt"
        text = "Тестовый текст для создания графа."
        dialog_id = "test_dialog_123"
        
        await load_service.process_file(file_id, file_name, text, dialog_id)
        
        # Проверяем, что neo.create_graph был вызван
        mock_neo.create_graph.assert_called_once()
        
        # Проверяем, что в Neo4j были переданы чанки
        call_args = mock_neo.create_graph.call_args[0][0]
        assert len(call_args) == 3
    
    @pytest.mark.asyncio
    async def test_process_file_metadata_correctness(self, load_service, mock_qdrant):
        """Тест: метаданные чанков корректны."""
        file_id = "test_file_123"
        file_name = "test_document.txt"
        text = "Тестовый текст."
        dialog_id = "test_dialog_456"
        
        await load_service.process_file(file_id, file_name, text, dialog_id)
        
        # Проверяем метаданные в переданных чанках
        call_args = mock_qdrant.load_in_qdrant.call_args[0][0]
        for chunk in call_args:
            assert chunk["file_id"] == file_id
            assert chunk["file_name"] == file_name
            assert chunk["dialog_id"] == dialog_id
            assert chunk["chunk_id"] is not None
            assert isinstance(chunk["chunk_id"], str)
            assert len(chunk["chunk_id"]) > 0
    
    @pytest.mark.asyncio
    async def test_process_files_multiple_files(self, load_service):
        """Тест: обработка нескольких файлов."""
        content = [
            {
                "fileId": "file_1",
                "fileName": "doc1.txt",
                "text": "Первый документ."
            },
            {
                "fileId": "file_2",
                "fileName": "doc2.txt",
                "text": "Второй документ."
            },
            {
                "fileId": "file_3",
                "fileName": "doc3.txt",
                "text": "Третий документ."
            }
        ]
        dialog_id = "test_dialog_789"
        
        await load_service.process_files(content, dialog_id)
        
        # Проверяем, что каждый файл был обработан
        # Каждый файл создает 3 чанка, всего 9 чанков
        assert load_service.embedder.embed.call_count == 9
        assert load_service.qdrant.load_in_qdrant.call_count == 3
        assert load_service.neo.create_graph.call_count == 3
    
    @pytest.mark.asyncio
    async def test_process_file_empty_text(self, load_service):
        """Тест: обработка файла с пустым текстом."""
        file_id = str(uuid.uuid4())
        file_name = "empty.txt"
        text = ""
        dialog_id = "test_dialog_empty"
        
        # Должно обработаться без ошибок
        await load_service.process_file(file_id, file_name, text, dialog_id)
        
        # Проверяем, что chunker был вызван даже с пустым текстом
        load_service.chunker.make_chunks_from_text.assert_called_once_with("")
    
    @pytest.mark.asyncio
    async def test_process_file_large_text(self, load_service):
        """Тест: обработка большого текста."""
        file_id = str(uuid.uuid4())
        file_name = "large_doc.txt"
        # Создаем большой текст (1000 символов)
        text = "Это большой документ. " * 50
        dialog_id = "test_dialog_large"
        
        await load_service.process_file(file_id, file_name, text, dialog_id)
        
        # Проверяем, что chunker был вызван с большим текстом
        load_service.chunker.make_chunks_from_text.assert_called_once_with(text)