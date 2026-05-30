from setuptools import setup, find_packages

setup(
    name="gatorrisk",
    version="1.0.0",
    description="Clinical NLP pipeline for lifestyle risk factor extraction from unstructured notes. "
                "Built at UF as an extension of the CTSI NLP Core GatorTron smoking extractor.",
    author="UF CISE / CTSI NLP Research",
    url="https://github.com/yourusername/gatorrisk",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "mlflow>=2.12.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
        "clinical": ["medspacy>=1.0.0", "spacy>=3.7.0"],
    },
    entry_points={
        "console_scripts": [
            "gatorrisk=modules.pipeline:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
