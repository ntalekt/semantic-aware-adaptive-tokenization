from setuptools import setup, find_packages

setup(
    name="sat-tokenizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.30.0",
        "sentencepiece>=0.1.99",
        "jellyfish>=0.11.2",
        "torch>=2.0.0",
        "scikit-learn>=1.2.0",
        "sentence-transformers>=2.2.2",
    ],
    author="ntalekt",
    author_email="ntalekt@users.noreply.github.com",
    description="Semantic-Aware Adaptive Tokenization for NLP tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ntalekt/semantic-aware-adaptive-tokenization",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
