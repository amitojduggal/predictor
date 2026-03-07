from setuptools import setup, find_packages

setup(
    name='imghdr-compatibility-fix',
    version='1.0',
    description='Provides a dummy imghdr module for Python 3.13+',
    packages=['imghdr'],
    package_dir={'imghdr': 'imghdr'},
)
