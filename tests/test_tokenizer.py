import unittest
from sat_tokenizer.tokenizer import SATokenizer


# tests/test_tokenizer.py
class TestSATokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SATokenizer(sp_model_path=None)

    def test_basic_tokenization(self):
        text = "New York City is amazing"
        result = self.tokenizer.tokenize(text)
        # Allow either merged or split tokens
        self.assertTrue(
            result
            in [
                ["New York City", "is", "amazing"],
                ["New", "York", "City", "is", "amazing"],
            ]
        )

    def test_special_characters(self):
        text = "GPT-4 outperformed state-of-the-art models!"
        result = self.tokenizer.tokenize(text)
        self.assertIn(result, [
            ['GPT-4', 'outperformed', 'state-of-the-art', 'models', '!'],
            ['GPT-4', 'outperformed', 'state-of-the-art', 'models!']
        ])

    def test_multilingual(self):
        text = "La inteligencia artificial está revolucionando el mundo"
        result = self.tokenizer.tokenize(text)
        self.assertTrue(len(result) >= 3)  # Check minimum token count

    def test_model_loading(self):
        embeddings = self.tokenizer._get_embeddings(["test"])
        self.assertEqual(embeddings.shape[1], 768)


if __name__ == "__main__":
    unittest.main()
