# Style and code guide for the CyberRisk project

## Language

As a rule, all code should be written in Python3 >= 3.8

Other languages is acceptable if it can be argued that not using Python will make the code easier to implement and/or maintain. The reason for chosing something other than Python must be OK with the project lead and documented.

## Header

All code must be contain a header with the name and email of the developers of
the code (more than one contributer, more than one name  ) All code must be

## Version Control

under version control
default version control is Git

## Code style

This section outlines code styles that should be followed for various languages.

Python

All code should PEP8 and generally splitting code up into multiple small functions is highly recommended.

We are using

* [Black](https://pypi.org/project/black/) to enforce code formatting.
* [isort](https://pycqa.github.io/isort/) to enforce import order
* [Flake8](https://flake8.pycqa.org/en/latest/) to enforce style guides.
* [pylint](https://pypi.org/project/pylint/) for code analysis/linting
* [mypy](https://mypy-lang.org/) for static typing (Optional, but highly recommended to catch trivial bugs)
