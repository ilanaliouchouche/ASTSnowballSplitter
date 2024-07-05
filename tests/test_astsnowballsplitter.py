import unittest
from src.astsnowballsplitter import ASTSnowballSplitter, ASTLanguageConfig
from transformers import AutoTokenizer
import os

current_dir = os.path.dirname(__file__)


class TestASTSnowballSplitter(unittest.TestCase):
    def setUp(self):
        languages_config = [
            ASTLanguageConfig(
                language='python',
                library_path=current_dir + '/build/my-languages.so',
                grammar_path=current_dir + '/tree-sitter-python'),
        ]
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.splitter = ASTSnowballSplitter(
            tokenizer=tokenizer,
            chunk_size=20,
            chunk_overlap=0,
            languages_config=languages_config
        )

    def test_split_text(self):
        python_code = """
def example_function():
    print("Hello, world!")
    for i in range(10):
        print(i)

def another_example_function_more_complex():
    print("Hello, world!")
    for i in range(10):
        print(i)
        if i % 2 == 0:
            print("Even number")
        else:
            print("Odd number")
        while i < 5:
            print("Less than 5")
            i += 1
        print("Greater than 5")
        if i == 5:
            print("Equal to 5")
        else:
            print("Not equal to 5")
        for j in range(3):
            print(j)
        print("End of loop")
    print("End of function")

def yet_another_example_function_but_recursive(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return yet_another_example_function_but_recursive(n - 1) +\\
            yet_another_example_function_but_recursive(n - 2)
"""

        texts = [python_code]
        file_extensions = ['py']
        documents = self.splitter.split_text(texts, file_extensions)

        self.assertGreater(len(documents), 0)
        for doc in documents:
            self.assertIn('file_extension', doc.metadata)


if __name__ == '__main__':
    unittest.main()
