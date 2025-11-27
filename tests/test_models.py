import unittest
from unittest.mock import patch, MagicMock
from models.ollama_model import OllamaChatModel
from models.openai_model import OpenAIChatModel

class TestOllamaModel(unittest.TestCase):
    @patch('models.ollama_model.requests.post')
    def test_generate(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Hello"}}
        mock_post.return_value = mock_response

        model = OllamaChatModel()
        response = model.generate([{"role": "user", "content": "Hi"}])
        self.assertEqual(response, "Hello")

    @patch('models.ollama_model.requests.post')
    def test_stream(self, mock_post):
        mock_response = MagicMock()
        lines = [
            b'{"message": {"content": "He"}}',
            b'{"message": {"content": "llo"}}'
        ]
        mock_response.iter_lines.return_value = lines
        mock_post.return_value.__enter__.return_value = mock_response

        model = OllamaChatModel()
        chunks = list(model.stream([{"role": "user", "content": "Hi"}]))
        self.assertEqual("".join(chunks), "Hello")

class TestOpenAIModel(unittest.TestCase):
    @patch('models.openai_model.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test'})
    def test_generate(self, mock_openai):
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "GPT response"
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        model = OpenAIChatModel()
        response = model.generate([{"role": "user", "content": "Hi"}])
        self.assertEqual(response, "GPT response")

if __name__ == '__main__':
    unittest.main()
