from setuptools import setup, find_packages

setup(
    name='GaMaDHaNi',  # Replace with your project's name
    version='0.1',
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[],       # List your project's dependencies
    include_package_data=True, # Include files specified in MANIFEST.in
    description='Official Codebase for the ISMIR 2024 paper; GaMaDHaNi: Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music',
    author='Nithya Shikarpur, Krishna Maneesha Dendukuri, Yusong Wu, Antoine Caillon, Cheng-Zhi Anna Huang',
    author_email='your.email@example.com',
    url='https://github.com/snnithya/GaMaDHaNi',
)