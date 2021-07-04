from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="tabular_anomaly_detection",
    version='0.0.1',
    author="Nathan Raw",
    author_email="naterawdata@gmail.com",
    install_requires=requirements,
    packages=find_packages(),
)
