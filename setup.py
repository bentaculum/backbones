from setuptools import setup

setup(
    name="backbones",
    version="0.1",
    description="Common neural network architectures implemented in PyTorch.",
    url="https://github.com/bentaculum/backbones",
    author="Benjamin Gallusser",
    author_email="benjamin.gallusser@epfl.ch",
    license="MIT",
    install_requires=[
        'pytest',
        'torch',
    ],
)
