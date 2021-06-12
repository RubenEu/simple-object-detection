#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'numpy', 'opencv-python', 'torch', 'torchvision', 'tqdm',
                'matplotlib', 'pandas', 'seaborn', 'tensorboard']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Rubén García Rojas",
    author_email='garcia.ruben@outlook.es',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Conjunto de herramientas y modelos para la detección de objetos.",
    entry_points={
        'console_scripts': [
            'simple_object_detection=simple_object_detection.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='simple_object_detection',
    name='simple_object_detection',
    packages=find_packages(include=['simple_object_detection', 'simple_object_detection.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/rubeneu/simple_object_detection',
    version='1.3.0',
    zip_safe=False,
)
