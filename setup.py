from setuptools import setup


setup(
    name='skits',
    version='0.1.0',
    description='scikit-inspired time series',
    author='Ethan Rosenthal',
    author_email='ethanrosenthal@gmail.com',
    license='MIT',
    url='https://github.com/EthanRosenthal/skits',
    packages=['skits'],
    test_suite='tests',
    install_requires=list(open('requirements.txt').read().strip().split('\n'))
)
