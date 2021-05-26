# A minimal setup.py file to make a Python project installable.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]


setuptools.setup(
    name             = "CI Bin",
    version          = "0.0.1",
    author           = "Armaan Kalyanpur, Olivia McNary, Andrew Kaplan, Mengdi Gao",
    author_email     = "armaankalyanpur@berkeley.edu",
    description      = "A Python library to test the efficacy of treatments, such as the Regeneron antibody cocktail's effect on COVID.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages         = setuptools.find_packages(),
    classifiers       = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires  = '>= 3.8',
    install_requires = requirements,
)123
