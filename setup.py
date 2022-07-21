from setuptools import setup

with open('requirements.txt', "r") as f:
    required = f.read().splitlines()

setup(
    name='pysven',
    version='0',
    packages=['pysven'],
    url="https://github.com/jameschapman19/pysven",
    license='MIT',
    requires=required,
    author='James Chapman',
    author_email='james.chapman.19@ucl.ac.uk',
    description='Elastic Net implementation using SVM solver'
)
