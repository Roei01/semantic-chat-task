import unittest
from unittest.mock import patch, MagicMock
from rag_service import LegalRAGService
from langchain_core.documents import Document

class TestRAGService(unittest.TestCase):
    @patch('rag_service.Chroma')
    @patch('rag_service.HuggingFaceEmbeddings')
    @patch('rag_service.VECTOR_DB_DIR')
    def test_retrieve(self, mock_dir, mock_embeddings, mock_chroma):
        mock_dir.exists.return_value = True
        
        mock_db_instance = MagicMock()
        mock_retriever = MagicMock()
        mock_db_instance.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_db_instance
        
        mock_retriever.invoke.return_value = [
            Document(page_content="Content 1", metadata={"filename": "doc1.pdf"}),
            Document(page_content="Content 2", metadata={"filename": "doc2.docx"})
        ]

        mock_chat = MagicMock()
        service = LegalRAGService(chat_model=mock_chat)
        
        docs = service.retrieve("test query")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["filename"], "doc1.pdf")

    @patch('rag_service.Chroma')
    @patch('rag_service.HuggingFaceEmbeddings')
    @patch('rag_service.VECTOR_DB_DIR')
    def test_answer(self, mock_dir, mock_embeddings, mock_chroma):
        mock_dir.exists.return_value = True
        mock_db_instance = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [Document(page_content="Context", metadata={"filename": "doc.pdf"})]
        mock_db_instance.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_db_instance

        mock_chat = MagicMock()
        mock_chat.generate.return_value = "Answer based on context"
        
        service = LegalRAGService(chat_model=mock_chat)
        answer, citations = service.answer("Question")
        
        self.assertEqual(answer, "Answer based on context")
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["filename"], "doc.pdf")

if __name__ == '__main__':
    unittest.main()
