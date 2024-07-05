# AST Snowball Splitter

AST Snowball Splitter is a Python package designed for intelligent code chunking using the nodes of Abstract Syntax Trees (AST). This approach allows for more relevant chunks compared to treating code as natural language. It is ideal for use in Retrieval-Augmented Generation (RAG) systems employed in coding assistants.

## Features

- **Intelligent Code Chunking**: Generates chunks based on the structure of the code, resulting in more meaningful segments.
- **AST-Based**: Utilizes Abstract Syntax Trees to split the code, ensuring that chunks respect the logical boundaries of the code.
- **RAG System Integration**: Perfect for use in Retrieval-Augmented Generation systems, enhancing the capabilities of coding assistants.
- **Langchain Integration**: Outputs are compatible with Langchain, using `Document` types for seamless integration.

## Installation

To install the package, use pip:

```bash
pip install astsnowballsplitter
```

## Usage

```python
from astsnowballsplitter.code_splitter import ASTSnowballSplitter, ASTLanguageConfig
from transformers import AutoTokenizer
from langchain.schema import Document

# Define the configuration for the languages
languages_config = [
    ASTLanguageConfig(language='python', library_path='build/my-languages.so', grammar_path='tree-sitter-python'),
    ASTLanguageConfig(language='javascript', library_path='build/my-languages.so', grammar_path='tree-sitter-javascript')
]

# Initialize the tokenizer and splitter
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
splitter = ASTSnowballSplitter(
    tokenizer=tokenizer,
    chunk_size=100,
    chunk_overlap=10,
    languages_config=languages_config
)

# Define source code to split
python_code = """
def example_function():
    print("Hello, world!")
    for i in range(10):
        print(i)
"""

javascript_code = """
function exampleFunction() {
    console.log("Hello, world!");
    for (let i = 0; i < 10; i++) {
        console.log(i);
    }
}
"""

# Split the code into chunks
texts = [python_code, javascript_code]
file_extensions = ['py', 'js']
documents = splitter.split_text(texts, file_extensions)

# Display the results
for doc in documents:
    print(doc.page_content)
    print(doc.metadata)
```
