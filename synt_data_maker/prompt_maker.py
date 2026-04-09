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

import re
import os
import logging
from abc import ABC


class PromptMaker(ABC):
    """
    Utility class for building prompts from templates with placeholder patterns.

    This class replaces placeholder substrings within a template using values
    defined in a dictionary. Placeholders follow a pattern such as ``"**-**"``,
    where the hyphen represents a variable gap.

    Parameters
    ----------
    prompt_template : str
        The template string containing placeholders to be filled.
    plh_replace_map_dict : dict
        A dictionary mapping placeholder keys to replacement text.
    gap_ph_pattern : str, optional
        The placeholder pattern indicating gaps, by default ``"**-**"``.
    root : str or None, optional
        Root directory path. If ``None``, the current working directory is used.

    Attributes
    ----------
    root : str
        The resolved root directory path.
    prompt_template : str
        The stored prompt template.
    plh_replace_map_dict : dict
        Mapping of keys to replacement values.
    gap_ph_pattern : str
        Pattern describing the placeholder format. It must contains a hypen.
    logger : logging.Logger
        Instance-specific logger for the class.
    """

    def __init__(
            self,
            prompt_template,
            plh_replace_map_dict,
            gap_ph_pattern='**-**',
            root=None
    ):
        self.set_logger()
        if root is None:
            self.root = os.path.abspath('.')
        else:
            self.root = root
        self.prompt_template = prompt_template
        self.plh_replace_map_dict = plh_replace_map_dict
        self.gap_ph_pattern = gap_ph_pattern

    def set_logger(self):
        """
        Configure an instance-level logger for the class.

        Notes
        -----
        - Prevents propagation to avoid duplicate logs.
        - Creates a stream handler with timestamped formatter.
        """
        self.logger = logging.getLogger(f'PromptMaker.{id(self)}')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(stream_handler)
        self.logger.propagate = False

    @staticmethod
    def check_text(string: str, text: str):
        """
        Validate that ``text`` exists within ``string``.

        Parameters
        ----------
        string : str
            The larger string to search.
        text : str
            The substring that must be present.

        Raises
        ------
        ValueError
            If ``text`` is not found within ``string``.
        """
        if text not in string:
            raise ValueError(f"'{text}' not found in {string}.")

    @staticmethod
    def fill_gap(string: str, pattern: str, text: str) -> str:
        """
        Replace the first occurrence of a placeholder pattern with a value.

        Parameters
        ----------
        string : str
            The string containing the placeholder.
        pattern : str
            The placeholder pattern to replace.
        text : str
            The replacement text.

        Returns
        -------
        str
            The string with the placeholder replaced.

        Raises
        ------
        ValueError
            If the placeholder pattern is not found.
        """
        PromptMaker.check_text(string, pattern)
        return string.replace(pattern, text)

    @staticmethod
    def get_re_pattern(pattern: str) -> str:
        """
        Convert a placeholder pattern into a regex pattern capturing the gap.

        Parameters
        ----------
        pattern : str
            Placeholder pattern, e.g., ``"**-**"``.

        Returns
        -------
        str
            A regex pattern where the hyphen is replaced with a capture group.
        """
        escaped = re.escape(pattern)
        regex = escaped.replace(r"\-", r"(.*?)")
        return regex

    @staticmethod
    def get_gap_keys(text: str, pattern: str) -> list:
        """
        Extract all placeholder keys found in the template.

        Parameters
        ----------
        text : str
            The template string.
        pattern : str
            The placeholder pattern.

        Returns
        -------
        list of str
            List of keys extracted from the template.

        Raises
        ------
        ValueError
            If the pattern does not contain a hyphen, which indicates the gap.
        """
        PromptMaker.check_text(pattern, '-')
        re_pattern = PromptMaker.get_re_pattern(pattern)
        return re.findall(re_pattern, text)

    @staticmethod
    def fill_text(template: str, text_dict: str, pattern="**-**") -> str:
        """
        Replace all placeholders in a template using a dictionary of values.

        Parameters
        ----------
        template : str
            The template containing one or more placeholders.
        text_dict : dict
            Mapping of placeholder keys to replacement values.
        pattern : str, optional
            Placeholder format, by default ``"**-**"``.

        Returns
        -------
        str
            Fully resolved text with all placeholders replaced.

        Raises
        ------
        KeyError
            If any placeholder key is missing in ``text_dict``.
        ValueError
            If a placeholder cannot be resolved.
        """
        gap_keys = PromptMaker.get_gap_keys(template, pattern)
        final_text = template
        for gap_key in gap_keys:
            gap = PromptMaker.fill_gap(pattern, '-', gap_key)
            final_text = PromptMaker.fill_gap(final_text, gap, text_dict[gap_key])
        return final_text

    def fill_prompt(self):
        """
        Fill the object's prompt template using the instance value dictionary.
        Returns
        -------
        str
            Rendered prompt with all placeholders replaced.

        Notes
        -----
        This is a convenience wrapper around :meth:`fill_text`.
        """
        filled_prompt = self.fill_text(
            self.prompt_template,
            self.plh_replace_map_dict,
            self.gap_ph_pattern
        )
        final_prompt = "\n".join(line.strip() for line in filled_prompt.splitlines() if line.strip())
        return final_prompt
