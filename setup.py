from setuptools import setup, find_packages

setup(
    name='GaMaDHaNi',  
    version='0.1',
    packages=find_packages(),  
    install_requires=[],       
    include_package_data=True, 
    description='Official Codebase for the ISMIR 2024 paper; GaMaDHaNi: Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music',
    author='Nithya Shikarpur, Krishna Maneesha Dendukuri, Yusong Wu, Antoine Caillon, Cheng-Zhi Anna Huang',
    author_email='snnithya@gmail.com',
    url='https://github.com/snnithya/GaMaDHaNi',
)