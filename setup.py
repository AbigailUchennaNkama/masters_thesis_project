from setuptools import setup
from setuptools import find_packages

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='thesis',
      description="package description",
      packages=find_packages(), # to find packages automatically
      install_requires=requirements)