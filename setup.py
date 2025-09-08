# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:04:30 2024

@author: git.metodologia@ine.es
"""

import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
VERSION = '0.1.0'
PACKAGE_NAME = 'codauto'
AUTHOR = 'Unidad de clasificaciones INE'
AUTHOR_EMAIL = 'git.metodologia@ine.es'
URL = 'https://github.com/AIML4OS/WP10_Cluster1_StatCodGen'
LICENSE = 'EUPL European Public License'  # Tipo de licencia
DESCRIPTION = 'Library for training an automatic coder'
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding='utf-8')  
LONG_DESC_TYPE = "Long description"
INSTALL_REQUIRES = [
    'pandas',
    'numpy',
    'pyreadstat',
    'utils',
    'fasttext-wheel',
    'openpyxl',
    'matplotlib',
    'scikit-learn',
    'nltk',
    'pyreadstat',
    'xlrd'
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True
)
