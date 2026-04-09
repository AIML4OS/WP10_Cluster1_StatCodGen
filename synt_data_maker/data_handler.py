# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Copyright (C) [2025] Instituto Nacional de Estadística
#
# Este archivo forma parte del proyecto statcodgen.
#
# Licenciado bajo la Licencia Pública de la Unión Europea (EUPL) v.1.2.
# Puede obtener una copia de la licencia en la raiz de este proyecto o en:
# https://eupl.eu/1.2/es/
#
# A menos que se indique lo contrario, este software se distribuye
# "TAL CUAL", SIN GARANTÍAS NI CONDICIONES DE NINGÚN TIPO.
# Consulte la licencia para conocer los términos específicos.
# ------------------------------------------------------------------------------
# Copyright (C) [2025] National Institute of Statistics
#
# This file is part of the statcodgen project.
#
# Licensed under the European Union Public License (EUPL) v.1.2.
# You can obtain a copy of the license at the root of this project or at:
# https://eupl.eu/1.2/es/
#
# Unless otherwise indicated, this software is distributed
# "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the license for specific terms.
# ------------------------------------------------------------------------------

import pickle
import json
import yaml


def read_json(name: str) -> dict:
    """
    Read a JSON file from disk.

    Parameters
    ----------
    name : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If `name` does not exist.
    json.JSONDecodeError
        If the file content is not valid JSON.
    UnicodeDecodeError
        If the file cannot be decoded as UTF-8.
    """
    with open(name, "r", encoding="utf-8") as j:
        data = json.load(j)
    return data


def save_json(name: str, data: dict):
    """
    Save a dictionary as a JSON file.

    Parameters
    ----------
    name : str
        Output path for the JSON file.
    data : dict
        Data to serialize as JSON.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the file cannot be written.
    TypeError
        If `data` contains non-JSON-serializable objects.

    Notes
    -----
    The current implementation uses the default `json.dump` settings (no
    indentation, ASCII escaping enabled by default).
    """
    with open(name, "w") as j:
        json.dump(data, j)


def read_pkl(name: str):
    """
    Load a Python object from a pickle file.

    Parameters
    ----------
    name : str
        Path to the pickle file.

    Returns
    -------
    object
        Deserialized Python object stored in the pickle file.

    Raises
    ------
    FileNotFoundError
        If `name` does not exist.
    pickle.UnpicklingError
        If the file is not a valid pickle stream.
    Exception
        Any exception raised during unpickling (e.g., missing class definitions).

    Warnings
    --------
    Loading pickle files can execute arbitrary code. Only unpickle data from
    trusted sources.
    """
    with open(name, 'rb') as p:
        data = pickle.load(p)
    return data


def save_pkl(name: str, data):
    """
    Serialize and save a Python object to a pickle file.

    Parameters
    ----------
    name : str
        Output path for the pickle file.
    data : object
        Python object to serialize.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the file cannot be written.
    pickle.PicklingError
        If `data` cannot be pickled.

    Notes
    -----
    Uses `pickle.HIGHEST_PROTOCOL`.
    """
    with open(name, 'wb') as p:
        pickle.dump(data, p, protocol=pickle.HIGHEST_PROTOCOL)


def read_yaml(name: str):
    """
    Read a YAML file from disk using safe loading.

    Parameters
    ----------
    name : str
        Path to the YAML file.

    Returns
    -------
    object
        Parsed YAML content (commonly a dict or list), depending on the YAML
        structure.

    Raises
    ------
    FileNotFoundError
        If `name` does not exist.
    yaml.YAMLError
        If the file content is not valid YAML.
    UnicodeDecodeError
        If the file cannot be decoded as UTF-8.
    """
    with open(name, 'r', encoding="utf-8") as y:
        data = yaml.safe_load(y)
    return data


def save_yaml(name:str, data):
    """
    Save data as a YAML file.

    Parameters
    ----------
    name : str
        Intended output path for the YAML file.
    data : object
        Data to serialize to YAML (typically dict, list, etc.).

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the file cannot be written.
    yaml.YAMLError
        If `data` cannot be represented as YAML.

    Notes
    -----
    The current implementation ignores the `name` parameter and always writes
    to `'data.yml'`.
    """
    with open('data.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

