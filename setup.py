### setup.py ###
from setuptools import setup, find_packages

setup(
    name='comparison-of-sr-methods',
    version='1.0.0',
    description='Comparision of different super-resolution methods',
    author='Lukas Hueglin',
    author_email='lukas.hueglin@outlook.com',
    packages=find_packages(include=[
        'dataset_creation',
        'super_resolution'
    ]),
    install_requires=[
        'numpy==1.22.3',
        'opencv-python==4.5.5.64',
        'matplotlib== 3.5.1',
        'pandas==1.4.1',
        'tqdm==4.63.1',
        'requests==2.27.1',
        'tensorflow==2.8.0',
        'h5py==3.6.0',
        'imageio==2.17.0',
        'GPUtil==1.4.0',
        'psutil==5.9.0'
    ]
)
