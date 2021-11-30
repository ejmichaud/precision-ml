# -*- coding: UTF-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='precisionml',
    version='0.0.1',
    author='Eric J. Michaud, Max Tegmark',
    author_email='ericjm@mit.edu',
    license='MIT',
    description='Tools for training neural networks to very high precision.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "torch"
    ],
    packages = ['precisionml'],
)

