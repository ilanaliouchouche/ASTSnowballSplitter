from typing import List, Dict, Any, Callable, Optional, Union, Generator
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
from transformers import AutoTokenizer
from tree_sitter import Language, Parser, Node
import warnings
import os
from dataclasses import dataclass

warnings.filterwarnings("ignore")


@dataclass
class ASTLanguageConfig:
    language: str
    library_path: str
    grammar_path: str


class ASTSnowballSplitter:
    """
    This class is used to split code documents into smaller chunks
    based on the Abstract Syntax Tree (AST) of the code. It's designed
    for RAG systems that index code documents and generate code completions
    based on the indexed documents. The algorithm is inspired from the one
    used in the continue plugin but enhanced with the ability to split the
    content instead of removing it.
    """

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 chunk_size: int,
                 chunk_overlap: int,
                 languages_config: List[ASTLanguageConfig],
                 map_lang_extension: Optional[Dict[str, str]] = None) -> None:
        """
        Constructor of the ASTSnowballSplitter class.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to use for tokenizing
                                       the text.
            chunk_size (int): The maximum number of tokens in each chunk.
            chunk_overlap (int): The number of tokens to overlap between
                                 consecutive chunks.
            languages_config (List[ASTLanguageConfig]):
                A list of configurations for each language to use.
                Each configuration must contain the following fields:
                    - language (str): The name of the language.
                    - library_path (str): The path to the language library.
                    - grammar_path (str): The path to the grammar files.
            map_lang_extension (Optional[Dict[str, str]]):
                A dictionary mapping language names to file extensions.
                If not provided, the default mapping is used.
        """

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.languages_config = languages_config
        self.map_lang_extension = map_lang_extension or {
            "python": "py",
            "java": "java",
            "cpp": "cpp",
            "c": "c",
            "javascript": "js",
            "typescript": "ts",
            "go": "go",
            "rust": "rs",
            "php": "php",
            "ruby": "rb",
            "swift": "swift",
            "kotlin": "kt",
            "scala": "scala",
            "shell": "sh",
            "r": "r",
            "julia": "jl",
            "haskell": "hs",
            "perl": "pl",
            "lua": "lua",
            "dart": "dart",
            "elixir": "ex",
            "clojure": "clj",
        }
        self.parsers = self._load_languages()

    def _load_languages(self) -> Dict[str, Parser]:
        """
        Load the language parsers for the languages specified in the
        languages_config field.

        Returns:
            Dict[str, Parser]: A dictionary mapping language names to
                               their corresponding parsers.
        """

        parsers = {}

        for config in self.languages_config:
            library_path = config.library_path
            grammar_path = config.grammar_path
            if not os.path.exists(grammar_path):
                raise FileNotFoundError(
                    f"Grammar path does not exist: {grammar_path}")
            if not os.path.isdir(grammar_path):
                raise ValueError(
                    f"Grammar path is not a directory: {grammar_path}")
            try:
                Language.build_library(library_path, [grammar_path])
            except Exception as e:
                raise Exception(
                    f"Failed to build language library for {config.language}:"
                    f" {e}")
            try:
                language = Language(library_path, config.language)
                parser = Parser()
                parser.set_language(language)
                parsers[config.language] = parser
            except Exception as e:
                print("Failed to load the language library for"
                      f"{config.language}: {e}")
                continue

        parsers_extension = {}
        for lang, parser in parsers.items():
            parsers_extension[self.map_lang_extension.get(lang, lang)] = parser

        return parsers_extension

    def _token_count(self,
                     text: Union[str, bytes]) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text (Union[str, bytes]): The text to count the tokens in.

        Returns:
            int: The number of tokens in the text.
        """

        return len(self.tokenizer.tokenize(
            text if isinstance(text, str) else text.decode('utf-8'),
            add_special_tokens=False
        ))

    def _collapsed_replacement(self,
                               node: Node) -> str:
        """
        Get the collapsed replacement for the given node.

        Args:
            node (Node): The node to get the collapsed replacement for.

        Returns:
            str: The collapsed replacement for the node.
        """

        return "{ ... }" if node.type == "statement_block" else "..."

    def _first_child(self,
                     node: Node,
                     grammar_name: Union[Any, List[Any]]) -> Optional[Node]:
        """
        Get the first child node of the given node that has the specified

        Args:
            node (Node): The node to get the first child node for.
            grammar_name (Union[Any, List[Any]]): The grammar name of the
                                                  child node to get.

        Returns:
            Optional[Node]: The first child node that has the specified
                            grammar name.
        """

        if isinstance(grammar_name, list):
            return next((child for child in node.children
                         if child.type in grammar_name), None)
        return next((child for child in node.children
                     if child.type == grammar_name), None)

    def _collapse_children(self,
                           node: Node,
                           code: str,
                           block_types: List[str],
                           collapse_types: List[str],
                           collapse_block_types: List[str],
                           max_chunk_size: int,
                           token_counter: Callable[[Union[str, bytes]], int]
                           ) -> str:
        """
        Collapse the children of the given node.

        Args:
            node (Node): The node to collapse its children.
            code (str): The code of the node.
            block_types (List[str]): The types of the blocks to collapse.
            collapse_types (List[str]): The types of the nodes to collapse.
            collapse_block_types (List[str]): The types of the blocks to
                                              collapse.
            max_chunk_size (int): The maximum number of tokens in each chunk.
            token_counter (Callable[[Union[str, bytes]], int]):
                A function to count the number of tokens in a text.

        Returns:
            str: The code of the node with the children collapsed.
        """

        code = code[:node.end_byte]
        block = self._first_child(node, block_types)
        collapsed_children = []

        if block:
            children_to_collapse = [child for child in block.children
                                    if child.type in collapse_types]
            for child in reversed(children_to_collapse):
                grandchild = self._first_child(child, collapse_block_types)
                if grandchild:
                    start = grandchild.start_byte
                    end = grandchild.end_byte
                    collapsed_child = code[child.start_byte:start] +\
                        self._collapsed_replacement(grandchild)
                    code = code[:start] +\
                        self._collapsed_replacement(grandchild) + code[end:]
                    collapsed_children.insert(0, collapsed_child)

        code = code[node.start_byte:]
        removed_child = False
        while token_counter(code) > max_chunk_size and collapsed_children:
            removed_child = True
            child_code = collapsed_children.pop()
            index = code.rfind(child_code)
            if index > 0:
                code = code[:index] + code[index + len(child_code):]

        if removed_child:
            lines = code.split("\n")
            first_whitespace_in_group = -1
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "":
                    if first_whitespace_in_group < 0:
                        first_whitespace_in_group = i
                else:
                    if first_whitespace_in_group - i > 1:
                        lines = lines[:i + 1] +\
                            lines[first_whitespace_in_group + 1:]
                    first_whitespace_in_group = -1
            code = "\n".join(lines)

        return code

    def _construct_class_definition_chunk(self,
                                          node: Node,
                                          code: str,
                                          max_chunk_size: int,
                                          token_counter: Callable[
                                              [Union[str, bytes]], int]
                                          ) -> str:
        """
        Construct a chunk for a class definition node.

        Args:
            node (Node): The class definition node.
            code (str): The code of the class definition node.
            max_chunk_size (int): The maximum number of tokens in each chunk.
            token_counter (Callable[[Union[str, bytes]], int]):
                A function to count the number of tokens in a text.

        Returns:
            str: The code of the class definition node with the children
                 collapsed.
        """

        return self._collapse_children(
            node, code, ["block", "class_body", "declaration_list"],
            ["function_definition", "function_declaration",
             "method_definition", "function_item"],
            ["block", "statement_block"], max_chunk_size, token_counter
        )

    def _split_function_or_method(self,
                                  node: Node,
                                  code: str,
                                  max_chunk_size: int,
                                  chunk_overlap: int,
                                  token_counter: Callable[
                                      [Union[str, bytes]], int]
                                  ) -> Generator[Dict[str, Any], None, None]:
        """
        Split a function or method node into smaller chunks.

        Args:
            node (Node): The function or method node to split.
            code (str): The code of the function or method node.
            max_chunk_size (int): The maximum number of tokens in each chunk.
            chunk_overlap (int): The number of tokens to overlap between
                                 consecutive chunks.

        Yields:
            Generator[Dict[str, Any], None, None]: A generator of chunks
                                                   containing the content,
                                                   start line, and end line.
        """

        body_node = node.children[-1]
        signature = code[node.start_byte:body_node.start_byte]
        body_text = code[body_node.start_byte:body_node.end_byte]

        if (
            node.parent
            and node.parent.type in ["block", "declaration_list"]
            and node.parent.parent
            and node.parent.parent.type in ["class_definition", "impl_item"]
           ):

            class_node = node.parent.parent
            class_block = node.parent
            signature = (
                f"{code[class_node.start_byte:class_block.start_byte]}...\n\n"
                f"{' ' * node.start_point[1]}{signature}"
            )

        splitter = TokenTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=max_chunk_size - token_counter(signature),
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(body_text)

        for i, chunk in enumerate(chunks):
            if i == 0:
                chunk_content = f"{signature}{chunk}\n..."
            elif i == len(chunks) - 1:
                chunk_content = f"{signature}\n...\n{chunk}"
            else:
                chunk_content = f"{signature}\n...\n{chunk}\n..."

            yield {
                "content": chunk_content,
                "start_line": node.start_point[0],
                "end_line": node.end_point[0]
            }

    def _get_smart_collapsed_chunks(self,
                                    node: Node,
                                    code: str,
                                    max_chunk_size: int,
                                    chunk_overlap: int,
                                    token_counter: Callable[
                                        [Union[str, bytes]], int],
                                    root: bool = True
                                    ) -> Generator[Dict[str, Any], None, None]:
        """
        Get smart collapsed chunks for the given node.

        Args:
            node (Node): The node to get the smart collapsed chunks for.
            code (str): The code of the node.
            max_chunk_size (int): The maximum number of tokens in each chunk.
            chunk_overlap (int): The number of tokens to overlap between
                                 consecutive chunks.
            token_counter (Callable[[Union[str, bytes]], int]):
                A function to count the number of tokens in a text.
            root (bool): A flag indicating whether the node is the root node.
        """

        if (
            (root or node.type in ["class_definition", "class_declaration",
                                   "impl_item", "function_definition",
                                   "function_declaration",
                                   "function_item"]) and
            token_counter(node.text) < max_chunk_size
           ):

            yield {
                "content": node.text,
                "start_line": node.start_point[0],
                "end_line": node.end_point[0]
            }
            return

        if node.type in ["class_definition", "class_declaration", "impl_item"]:
            yield {
                "content": self._construct_class_definition_chunk(
                    node, code, max_chunk_size, token_counter),
                "start_line": node.start_point[0],
                "end_line": node.end_point[0]
            }
        elif node.type in ["function_definition", "function_declaration",
                           "function_item"]:
            yield from self._split_function_or_method(
                node, code, max_chunk_size, chunk_overlap, token_counter)

        for child in node.children:
            yield from self._get_smart_collapsed_chunks(
                child, code, max_chunk_size, chunk_overlap, token_counter,
                False)

    def split_document(self,
                       documents: List[Document]) -> List[Document]:
        """
        Split the given documents into smaller chunks.

        Args:
            documents (List[Document]): The documents to split.

        Returns:
            List[Document]: The list of chunks generated from the documents.
        """

        all_chunks = []
        for document in documents:
            file_extension = document.metadata.get('file_extension', '')
            if file_extension not in self.parsers:
                continue
            code = document.page_content
            metadata = document.metadata
            parser = self.parsers[file_extension]
            tree = parser.parse(bytes(code, 'utf8'))
            for chunk in self._get_smart_collapsed_chunks(tree.root_node,
                                                          code,
                                                          self.chunk_size,
                                                          self.chunk_overlap,
                                                          self._token_count):
                all_chunks.append(self._process_chunk(chunk,
                                                      metadata,
                                                      self._token_count))
        return all_chunks

    def split_text(self,
                   texts: List[str],
                   file_extensions: List[str]) -> List[Document]:
        """
        Split the given texts into smaller chunks.

        Args:
            texts (List[str]): The texts to split.
            file_extensions (List[str]): The file extensions of the texts.

        Returns:
            List[Document]: The list of chunks generated from the texts.
        """

        if len(texts) != len(file_extensions):
            raise ValueError("The number of texts must match "
                             "the number of file extensions")

        all_chunks = []
        for text, file_extension in zip(texts, file_extensions):
            doc = Document(page_content=text,
                           metadata={"file_extension": file_extension})
            chunks = self.split_document([doc])
            all_chunks.extend(chunks)

        return all_chunks

    def _process_chunk(self,
                       chunk: Dict[str, Any],
                       metadata: Dict[str, Any],
                       token_counter: Callable[
                           [Union[str, bytes]], int]
                       ) -> Document:
        """
        Process the given chunk and return a Document object.

        Args:
            chunk (Dict[str, Any]): The chunk to process.
            metadata (Dict[str, Any]): The metadata of the chunk.
            token_counter (Callable[[Union[str, bytes]], int]):
                A function to count the number of tokens in a text.

        Returns:
            Document: The Document object representing the chunk.
        """

        page_content = (
            chunk['content'].decode('utf-8')
            if isinstance(chunk['content'], bytes)
            else chunk['content']
            )
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "start_line": chunk['start_line'] if chunk['start_line'] else 0,
            "end_line": chunk['end_line'] if chunk['end_line'] else None,
            "number_tokens": token_counter(page_content),
        })
        return Document(page_content=page_content, metadata=chunk_metadata)
