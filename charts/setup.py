#!/usr/bin/env python

# packaging library
from setuptools import setup, find_packages

setup(name="chart-indicators",
      version="1.0",
      description="A python tool downloading data from the yahoo finance api and calculating price indicators.",
      author="Tobias Haider",
      author_email="tobias.haider99@hotmail.com",
      url="https://github.com/tobsel7/price-indicators",
      packages=find_packages(),
      )
