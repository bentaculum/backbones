from setuptools import setup, find_packages

setup(
    name="backbones",
    version="0.1",
    description="Common neural network architectures implemented in PyTorch.",
    url="https://github.com/bentaculum/backbones",
    author="Benjamin Gallusser",
    author_email="benjamin.gallusser@epfl.ch",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        'pytest',
        'torch',
    ],
)
