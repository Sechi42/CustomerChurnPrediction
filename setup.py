import io
import os
from pathlib import Path

from setuptools import find_packages, setup

# Metadata of package
NAME = 'CustomerChurnPrediction'
DESCRIPTION = 'Reviews Prediction Model'
URL = 'https://github.com/Sechi42/CustomerChurnPrediction'
EMAIL = 'sergio.anayads@gmail.com'
AUTHOR = 'Sergio Anaya'
REQUIRES_PYTHON = '>=3.12.3'

pwd = os.path.abspath(os.path.dirname(__file__))

# Get the list of packages to be installed
def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd, fname), encoding='utf-8') as f:
        return f.read().splitlines()

try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
about = {}
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={NAME: ['VERSION']},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license='Tripleten',  # Tu licencia personalizada
    classifiers=[
        'License :: Other/Proprietary License',  # Clasificador para una licencia personalizada
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
