import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="neurodata",
    version="0.1.0",
    description="Preprocess and load data from neuromorphic datasets",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/kclip/neurodata",
    author="Nicolas Skatchkovsky",
    author_email="office@realpython.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["neurodata"],
    include_package_data=True,
    install_requires=["torch", "numpy", "tables"]
)
