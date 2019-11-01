import os
from setuptools import setup, find_packages

from skits import __version__

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

with open(os.path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

with open(os.path.join(here, 'test-requirements.txt')) as f:
    test_requirements = f.read().splitlines()


setup(
    name='skits',
    version=__version__,
    description='scikit-learn-inspired time series',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ethan Rosenthal',
    author_email='ethanrosenthal@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='machine learning time series forecasting',
    license='MIT',
    url='https://github.com/EthanRosenthal/skits',
    packages=find_packages(exclude=['tests']),
    test_suite='tests',
    install_requires=requirements,
    extras_requires={
        'test': test_requirements
    }
)
