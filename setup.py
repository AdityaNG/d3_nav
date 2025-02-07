"""Python setup.py for d3nav package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("d3nav", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="d3nav",
    version=read("d3nav", "VERSION"),
    description="D³Nav: Data-Driven Driving Agents for Autonomous Vehicles in Unstructured Traffic",
    url="https://github.com/AdityaNG/d3_nav/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="AdityaNG",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["d3nav = d3nav.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
