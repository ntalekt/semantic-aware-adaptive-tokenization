import unittest
from sat_tokenizer.tokenizer import SATokenizer  # Updated import path


class TestSATokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SATokenizer()

    def test_basic_tokenization(self):
        text = "New York City is amazing"
        result = self.tokenizer.tokenize(text)
        self.assertEqual(result, ["New York City", "is", "amazing"])

    def test_special_characters(self):
        text = "GPT-4 outperformed state-of-the-art models!"
        result = self.tokenizer.tokenize(text)
        self.assertEqual(
            result, ["GPT-4", "outperformed", "state-of-the-art", "models", "!"]
        )

    def test_multilingual(self):
        text = "La inteligencia artificial está revolucionando el mundo"
        result = self.tokenizer.tokenize(text)
        self.assertEqual(
            result, ["La inteligencia artificial", "está revolucionando", "el mundo"]
        )


if __name__ == "__main__":
    unittest.main()
