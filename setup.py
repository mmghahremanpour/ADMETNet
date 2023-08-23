from setuptools import setup, find_packages

setup(
    name="ADMETNet",
    version="1.0.0",
    description="A Neural Network Framework for AMDET Prediction.",
    author="Mohammad M. Ghahremanpour",
    author_email="mohammad.ghahremanpour@yale.edu",
    url = "https://github.com/mmghahremanpour/ADMETNet.git",
    license="Apache License V2.0",
    classifiers=[
        "Development Status :: Production",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "solnet=main.solnet:solnet",
        ],
    },
)

