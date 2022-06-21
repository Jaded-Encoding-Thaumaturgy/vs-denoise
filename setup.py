#!/usr/bin/env python3

import setuptools

with open("requirements.txt") as fh:
    install_requires = fh.read()

name = "vs-denoise"
version = "0.0.0"

setuptools.setup(
    name=name,
    version=version,
    author="Irrational Encoding Wizardry",
    author_email="wizards@encode.moe",
    description="Vapoursynth denoising functions",
    packages=["vsdenoise"],
    url="https://github.com/Irrational-Encoding-Wizardry/vs-denoise",
    package_data={
        'vsdenoise': ['py.typed'],
    },
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
