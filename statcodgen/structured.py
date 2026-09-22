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
"""
Created on Thu Mar 14 11:04:48 2024

@author: git.metodologia@ine.es
"""
from abc import ABC, abstractmethod
import re
import pandas as pd

from .my_utils import select_query_sparql


class Structured(ABC):
    """
    Abstract base class for representing and processing hierarchical classification structures. 

    This class defines a general framework for managing hierarchical
    classification systems, such as taxonomies or multi-level coding schemes.
    It provides functionality to load the structure from either a DataFrame
    or a SPARQL endpoint, organize it hierarchically, and extract useful
    mappings between codes, titles, and hierarchical paths.

    Subclasses must implement the abstract method `get_level()`, which defines
    how to compute or interpret the level of a given classification code.
    Returned levels must be consecutive integers starting at 0, where
    0 represents the highest hierarchy level.

    Parameters
    ----------
    structure_df : pandas.DataFrame, optional
        Input DataFrame containing at least two columns:
        - The first column: classification codes (as strings, possibly with dots).
          Parent codes must appear before their child codes to preserve hierarchy.
        - The second column: corresponding classification titles.
        Used when `source='dataframe'`.
    sparql_endpoint : str, optional
        URL of the SPARQL endpoint to query. Used when `source='sparql_endpoint'`.
    query : str, optional
        SPARQL query string used to retrieve the classification structure.
        Must return columns compatible with `load_structure_from_sparql_endpoint()`.
    names_l : list of str, optional
        Optional list of human-readable names for each hierarchy level,
        ordered by increasing depth. If not provided, numeric indices are used.
    source : {'dataframe', 'sparql_endpoint'}, default='dataframe'
        Defines the source from which to load the structure:
        - `'dataframe'`: load directly from a pandas DataFrame.
        - `'sparql_endpoint'`: query a remote SPARQL endpoint.
    ensure_garbage_class : bool
        If True, garbage classes are automatically added to the
        structure following the convention:

        __G0__, __G1__, ..., __Gn__

        Models intended to be used with native filtering must be
        trained using a structure created with
        ``ensure_garbage_class=True``.

    Attributes
    ----------
    struc_df : pandas.DataFrame
        DataFrame containing classification codes, titles, and their computed hierarchy levels.
    reversed_hierarchy : dict
        Dictionary mapping each leaf code to its full hierarchical path (from root to leaf).
    titles : dict
        Mapping of classification codes (without dots) to their corresponding titles.
    up_codes_l : list of str
        Temporary container used during hierarchy construction to track parent-child chains.
    max_level : int
        Maximum depth (i.e., number of hierarchical levels) in the structure.
    level_dict : dict
        Mapping of level indices (integers) to their human-readable names, if provided.
        If `names_l` is not given, numeric indices are used as names.
    level_l : list of str or None
        List of names for each level in the hierarchy, corresponding to `level_dict.values()`.
  


    Raises
    ------
    ValueError
        If `source` is not `'dataframe'` or `'sparql_endpoint'`.
        If `names_l` is provided but its length does not match the number of detected levels.

    Notes
    -----
    - This class is intended to be subclassed by specific classification
      systems (e.g., industry codes, ontology structures, etc.).
    - The loading mechanism is determined at instantiation based on the
      `source` parameter.
    - The `get_reversed_hierarchy_and_titles()` method is executed at
      initialization to populate `reversed_hierarchy` and `titles`.

    Examples
    --------
    Load from a pandas DataFrame
    >>> df = pd.DataFrame([
    ...     ['A', 'Root'],
    ...     ['A.1', 'Child 1'],
    ...     ['A.2', 'Child 2']
    ... ])
    >>> struct = MyStructuredSubclass(structure_df=df, source='dataframe')
    >>> struct.max_level
    1

    Load from a SPARQL endpoint
    >>> struct = MyStructuredSubclass(
    ...     sparql_endpoint="http://example.org/sparql",
    ...     query="SELECT ?code ?title ?parent WHERE {...}",
    ...     source="sparql_endpoint"
    ... )
    """

    def __init__(
            self,
            structure_df=None,
            sparql_endpoint=None,
            query=None,
            names_l=None,
            source='dataframe',
            ensure_garbage_class = False
            ):
        if source == 'dataframe':
            self.struc_df = self.load_structure_from_dataframe(structure_df)
        elif source == 'sparql_endpoint':
            self.struc_df = self.load_structure_from_sparql_endpoint(
                sparql_endpoint,
                query
            )
        else:
            raise ValueError('Source must be "dataframe" or "sparql_endpoint"')
        levels = sorted(self.struc_df.iloc[:, 2].unique())
        if levels != list(range(len(levels))):
            raise ValueError(
                "Levels returned by get_level() must be consecutive integers starting at 0."
            )
        self.reversed_hierarchy = {}
        self.titles = {}
        self.up_codes_l = []
        self.max_level = max(self.struc_df.iloc[:, 2].unique())
        self.level_dict = self.get_name_dict(names_l)
        if names_l is None:
            self.level_l = sorted(list(self.level_dict.values()))
        else:
            if len(names_l) != self.max_level + 1:
                raise ValueError(
                    'names_l must have the same number of items as the structure has levels.'
                )
            self.level_l = names_l
        if ensure_garbage_class:
            self.create_garbage_class()
        self.get_reversed_hierarchy_and_titles()
    
    def create_garbage_class(self):
        """
        Append one garbage node per hierarchical level.

        Garbage codes follow the reserved pattern:

            __G0__
            __G1__
            ...
            __Gn__

        Levels are assigned explicitly when creating the rows and are
        therefore independent from the classification-specific implementation
        of ``get_level()``.

        Codification models must be trained using these labels in order to support native filtering.
        """

        code_n = self.struc_df.columns[0]
        title_n = self.struc_df.columns[1]
        level_n = self.struc_df.columns[2]

        new_rows = []

        for level in range(self.max_level + 1):
            new_rows.append({
                code_n: f'__G{level}__',
                title_n: 'Garbage',
                level_n: level
            })

        self.struc_df = pd.concat(
            [self.struc_df, pd.DataFrame(new_rows)],
            ignore_index=True
        )


    def load_structure_from_sparql_endpoint(self, sparql_endpoint, query):
        """
        Load and organize a hierarchical structure from a SPARQL endpoint.

        This method executes a predefined SPARQL query, retrieves the results into
        a DataFrame, and organizes the records into a hierarchical order based on
        their parent-child relationships. The method also normalizes the codes by
        removing period characters ('.') and returns only the relevant columns.
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the hierarchical structure with the following
            columns:
            - 'code': str, the identifier of each element (without dots)
            - 'title': str, the descriptive label or title
            - 'level': int, the hierarchy level (0 for root elements)

        Notes
        -----
        Steps performed:
        1. Executes the SPARQL query and loads the result as a DataFrame.
        2. Sorts and organizes the hierarchy using the `sort_hierarchy()` method.
        3. Cleans the 'code' field by removing '.' characters.
        4. Returns the processed DataFrame.
        """
        structure_df = select_query_sparql(
            sparql_endpoint, query
        )
        rows = self.sort_hierarchy(structure_df)
        sorted_structure_df = pd.DataFrame(rows)
        sorted_structure_df['code'] = sorted_structure_df['code'].apply(
            lambda code:
                code.replace('.', '')
        )
        return sorted_structure_df[['code', 'title', 'level']]

    @staticmethod
    def sort_hierarchy(df, parent=None, level=0):
        """
        Recursively sort a DataFrame based on parent-child hierarchy.

        This method traverses a DataFrame representing hierarchical relationships,
        recursively organizing the data into a list of dictionaries that include
        hierarchy levels. It identifies root nodes (where 'parent' is NaN) and then
        processes each child node in ascending order by 'code'.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing three columns: ['code', 'title', 'parent'].
            Each row represents a node, its title, and its parent code.
        parent : str or None, optional
            The parent code to build the hierarchy from. If None, the function
            starts at the root level. Default is None.
        level : int, optional
            The current hierarchy depth. Root elements are level 0. Default is 0.

        Returns
        -------
        list of dict
            A list of dictionaries, where each dictionary represents a node with
            the following keys:
            - 'code': str, the node identifier
            - 'title': str, the node label or name
            - 'parent': str or None, the parent node identifier
            - 'level': int, the hierarchical depth of the node

        Notes
        -----
        The function uses recursion to navigate the full hierarchy. It is
        typically used internally by `load_structure_from_sparql_endpoint()`.
        """
        df.columns = ['code', 'title', 'parent']
        if parent is None:
            children = df[df['parent'].isna()]
        else:
            children = df[df['parent'] == parent]
        rows = []
        for _, row in children.sort_values('code').iterrows():
            row_dict = row.to_dict()
            row_dict['level'] = level
            rows.append(row_dict)
            rows.extend(Structured.sort_hierarchy(df, row['code'], level + 1))
        return rows

    def load_structure_from_dataframe(self, struc_df):
        """
        Load the input DataFrame and compute the hierarchical level
        for each classification code.

        Parameters
        ----------
        struc_df : pandas.DataFrame
            Input DataFrame containing codes and titles.

        Returns
        -------
        pandas.DataFrame
            Copy of the input DataFrame with an additional column 'level'
            representing the computed hierarchical depth of each code.
        """
        out_df = struc_df.copy()
        out_df.insert(
            loc=2,
            column='level',
            value=out_df.iloc[:, 0].apply(
                self.get_level
            )
        )
        return out_df

    def get_name_dict(self, names_l):
        """
        Generate a dictionary mapping level indices to names.

        Parameters
        ----------
        names_l : list of str or None
            List of names for each hierarchy level. If None,
            numeric indices are used instead.

        Returns
        -------
        dict
            Dictionary mapping level indices to names or numeric values.
        """
        if names_l is not None:
            names_dict = dict(enumerate(names_l))
        else:
            names_dict = {
                idx: str(val)
                for idx, val in enumerate(range(self.max_level+1))
            }
        return names_dict

    def get_up_codes_list(self, code, level):
        """
        Update the chain of parent codes up to the given level.

        Parameters
        ----------
        code : str
            Classification code (without dots).
        level : int
            Hierarchical level of the code.
        """
        self.up_codes_l = self.up_codes_l[:level]
        self.up_codes_l.append(code)

    def get_reversed_hierarchy_and_titles(self):
        """
        Build the reversed hierarchy and title mappings.

        This method populates:
        - `titles`: mapping from code to title.
        - `reversed_hierarchy`: mapping from leaf codes to their full
          hierarchical path represented as a dictionary.
        """
        code_n = self.struc_df.columns[0]
        title_n = self.struc_df.columns[1]
        level_n = self.struc_df.columns[2]
        for _, row in self.struc_df.iterrows():
            title = row[title_n]
            code = row[code_n].replace('.', '')
            level = row[level_n]
            self.titles[code] = title
            if level == self.max_level:
                self.get_up_codes_list(code, level)
                self.reversed_hierarchy[code] = {
                    self.level_dict[index]: code
                    for index, code in enumerate(self.up_codes_l)
                }
            else:
                self.get_up_codes_list(code, level)

    @abstractmethod
    def get_level(self, code):
        """
        Abstract method to compute the hierarchical level of a given code.

        Notes
        -----
        Implementations must return hierarchy levels as consecutive integers
        starting at 0, where 0 represents the highest level of the hierarchy.

        Parameters
        ----------
        code : str
            Classification code (string, possibly with dots).

        Returns
        -------
        int
            The hierarchical level corresponding to the code.
        """


class StructuredCNAE(Structured):
    """
    Structured hierarchy for CNAE (Clasificación Nacional de Actividades
    Económicas).

    Levels are determined by the length of the classification code
    (after removing dots).
    """

    def get_level(self, code):
        """
        Compute the hierarchical level for a CNAE code.

        Parameters
        ----------
        code : str
            CNAE classification code.

        Returns
        -------
        int
            Hierarchical level based on code length minus one.
        """
        code = code.replace('.', '')
        return len(code)-1


class StructuredCNED(Structured):
    """
    Structured hierarchy for CNED (Clasificación Nacional de Educación).

    Levels are determined by both code length and whether the code
    contains only digits.  
    """

    def get_level(self, code):
        """
        Compute the hierarchical level for a CNED code.

        Parameters
        ----------
        code : str
            CNED classification code.

        Returns
        -------
        int
            Hierarchical level based on a combination of code length
            and digit-only validation.
        """
        code = code.replace(".", "")
        len_code = len(code)

        if len_code == 1:
            return 1 if code.isdigit() else 0
        elif len_code == 2:
            return 3 if code.isdigit() else 2
        else:
            raise ValueError(
                f"Invalid CNED code: {code}"
            )


class StructuredCNO(Structured):
    """
    Structured hierarchy for CNO (Clasificación Nacional de Ocupaciones).

    Levels are assigned based on code length and whether the code
    contains only digits.
    """

    def get_level(self, code):
        """
        Compute the hierarchical level for a CNO code.

        Parameters
        ----------
        code : str
            CNO classification code.

        Returns
        -------
        int
            Hierarchical level derived from code length and numeric validation.
        """
        code = code.replace('.', '')

        if len(code) == 1:
            return 0 if code.isdigit() else 1
        else:
            return len(code)


class StructuredCPA(Structured):
    """
    Structured hierarchy for CPA (Clasificación de Productos por Actividad).

    Levels are determined strictly by the length of the classification code.
    """

    def get_level(self, code):
        """
        Compute the hierarchical level for a CPA code.

        Parameters
        ----------
        code : str
            CPA classification code.

        Returns
        -------
        int
            Hierarchical level based on code length minus one.
        """
        code = code.replace('.', '')
        return len(code)-1

    
