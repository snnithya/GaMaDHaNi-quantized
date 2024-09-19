from setuptools import setup, find_packages

setup(
    name='GaMaDHaNi',  
    version='0.1',
    packages=find_packages(),  
    install_requires=[
        "x_transformers==1.30.2",
        "pytorch_lightning==1.9.0",
        "gin_config==0.5.0",
        "joblib==1.2.0",
        "numpy==1.24.4",
        "pytorch_lightning==1.9.0",
        "scikit_learn==1.2.0",
        "setuptools==67.8.0",
        "torch==2.4.0",
        "torchaudio==2.4.0",
        'tqdm==4.65.0',
        "matplotlib==3.4.3",
        "absl_py==1.4.0"
    ],       
    extras_require={
        'training': [
            "lmdb==1.4.1",
            "pandas==2.0.3",
            "protobuf==3.20.3",
            "wandb==0.15.4"
        ]
    },
    entry_points={
      'console_scripts': [
        'generate' = 'gamadhani.scripts.generate:main',
      ],
    },
    include_package_data=True, 
    description='Official Codebase for the ISMIR 2024 paper; GaMaDHaNi: Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music',
    author='Nithya Shikarpur, Krishna Maneesha Dendukuri, Yusong Wu, Antoine Caillon, Cheng-Zhi Anna Huang',
    author_email='snnithya@gmail.com',
    url='https://github.com/snnithya/GaMaDHaNi',
)