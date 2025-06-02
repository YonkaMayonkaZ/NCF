from setuptools import setup, find_packages

setup(
    name="ncf-kd",
    version="0.1.0", 
    packages=find_packages(),
    python_requires=">=3.6,<3.8",
    install_requires=[
        "torch==1.0.1",  
        "numpy==1.16.2",
        "pandas==0.24.2",
        "scipy",
        "matplotlib==3.3.4",
        "pyyaml>=3.13,<7.0",
        "tqdm>=4.0"
    ]
)
