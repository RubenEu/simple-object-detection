from setuptools import setup, find_packages

setup(
    name='simple-object-detection',
    version='0.1.0',
    description='Wrapper for the use of object detector with different models.',
    author='Rubén García Rojas',
    author_email='garcia.ruben@outlook.es',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'requests',
        'opencv-python',
        'torch',
        'torchvision',
        'PyYAML',
        'matplotlib',
        'pandas',
        'pillow',
        'tqdm',
        'seaborn',
    ]
)
