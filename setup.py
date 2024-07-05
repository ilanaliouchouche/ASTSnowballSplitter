from setuptools import setup, find_packages

setup(
    name='astsnowballsplitter',
    version='0.2.0',
    description='A package for smartly splitting code into chunks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Ilan Aliouchouche',
    author_email='ilan.aliouchouche@universite-paris-saclay.fr',
    url='https://github.com/ilanaliouchouche/ASTSnowballSplitter',
    packages=find_packages(),
    install_requires=[
        'tiktoken>=0.7.0',
        'transformers>=4.41.2',
        'langchain>=0.2.3',
        'spacy>=3.7.5',
        'tree_sitter>=0.20.1',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
