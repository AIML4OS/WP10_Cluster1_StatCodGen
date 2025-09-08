# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:04:48 2024

@author: git.metodologia@ine.es
"""
from abc import ABC, abstractmethod
import re


class Structured(ABC):
    """
    Abstract base class for representing and processing hierarchical
    classification structures.

    This class provides the general framework for loading, organizing,
    and traversing classification codes that form a tree-like hierarchy.
    Subclasses must implement the `get_level` method, which defines
    how to compute the hierarchical level of a given code.

    Parameters
    ----------
    structure_df : pandas.DataFrame
        DataFrame containing at least two columns:
        - The first column: classification codes (string values, possibly with dots).
          Codes must be provided in hierarchical order, meaning that parent classes
          must appear before their children.
        - The second column: classification titles.
    names_l : list of str, optional
        Optional increasing depth, sorted list of human-readable names for each level
        in the hierarchy. If not provided, levels are indexed numerically.

    Attributes
    ----------
    struc_df : pandas.DataFrame
        DataFrame containing the classification codes, titles,
        and computed levels.
    reversed_hierarchy : dict
        Mapping of leaf classification codes to their full hierarchical path.
    titles : dict
        Mapping of classification codes (without dots) to their titles.
    up_codes_l : list of str
        Temporary list storing the chain of parent codes while building
        the hierarchy.
    max_level : int
        Maximum depth (level) in the classification structure.
    level_dict : dict
        Mapping of level indices to their names (or numeric indices
        if `names_l` is not provided).
    level_l : list or None
        List of names assigned to hierarchy levels (or numeric indices
        if `names_l` is not provided).
    """

    def __init__(self, structure_df, names_l=None):
        self.struc_df = self.load_structue(structure_df)
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
        self.get_reversed_hierarchy_and_titles()

    def load_structue(self, struc_df):
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
            value=out_df.iloc[:, 0].apply(self.get_level)
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
                idx: str(val) for idx, val in enumerate(range(self.max_level+1))
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
        for index, row in self.struc_df.iterrows():
            title = row[title_n]
            code = row[code_n].replace('.', '')
            level = row[level_n]
            self.titles[code] = title
            if level == self.max_level:
                self.get_up_codes_list(code, level)
                self.reversed_hierarchy[code] = {
                    self.level_dict[index]: code for index, code in enumerate(self.up_codes_l)
                }
            else:
                self.get_up_codes_list(code, level)

    @abstractmethod
    def get_level(self, code):
        """
        Abstract method to compute the hierarchical level of a given code.

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
    Structured hierarchy for CNAE (Clasificación Nacional de Actividades Económicas).

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
        code = code.replace('.', '')
        len_code = len(code)
        bool_num = bool(re.fullmatch(r"\d+", code))
        if len_code == 1:
            if bool_num:
                return 1
            return 0
        if len_code == 2:
            if bool_num:
                return 3
            return 2
        return None


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
        len_code = len(code)
        if len_code == 1:
            if bool(re.fullmatch(r"\d+", code)):
                return 0
            return 1
        return len_code


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
