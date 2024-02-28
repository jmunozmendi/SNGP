from setuptools import setup, find_packages

setup(
    name="SNGP",
    version="0.4",
    packages=find_packages(),
    description="Spectral-normalized Neural Gaussian processes in PyTorch.",
    long_description=open('README.md').read(),
    author="Javier Muñoz Mendi",
    author_email="jmunozmendi@gmail.com",
    license="MIT",
    install_requires=[
        "numpy",
        "torch",  
    ],
)