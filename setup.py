# coding=utf-8

from distutils.core import setup

setup(
    # Application name:
    name="hof",

    # Version number (initial):
    version="0.1.0",

    # Application author details:
    author="Fábio Uechi",
    author_email="fabio.uechi@gmail.com",

    # Packages
    packages=[
        "hof",
    ],

    # Include additional files into the package
    include_package_data=True,

    # Details
    url="http://pypi.python.org/pypi/hof/",
    license="LICENSE",
    description="House of faces.",

    # long_description=open("README.txt").read(),

    # Dependent packages (distributions)
    install_requires=[
        'requests',
        'opencv-python'
    ],

    extras_require={
        'tensorflow_export': ["tensorflow"],
    }
)
