from setuptools import setup, find_packages

setup(
   name="syndatasetgen",
   version="1.0.0",
   author="Mark Fisher",
   author_email='mark.fisher@uea.ac.uk',
   description="synthetic-dataset-generator",
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown',
   url="https://git.imr.no/everyfish/synthetic-dataset-generator",
   packages=find_packages(),
   install_requires=[],
   license="GPLv3"
)