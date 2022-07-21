from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pysven',
    version='0',
    packages=['pysven'],
    url='',
    license='',
    requires=required,
    author='James Chapman',
    author_email='james.chapman.19@ucl.ac.uk',
    description='Elastic Net implementation using SVM solver'
)
