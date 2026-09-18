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

from abc import ABC
import re
import logging
import random

class NotesProcess(ABC):
    """
    Initialize a notes-processing pipeline.

    The initializer stores the input notes and examples data, optionally
    samples the notes DataFrame, initializes the internal dictionary used
    to accumulate per-code metadata, and configures an instance logger.

    Parameters
    ----------
    notes_df : pandas.DataFrame or None, optional
        DataFrame containing the notes to process. If `sample` is provided
        and truthy, a random sample is taken from this DataFrame.
    sample : int or None, optional
        Number of rows to sample from `notes_df`. If falsy/None, the full
        DataFrame is used.
    random_state : int, default=23
        Random seed used when sampling `notes_df`.
    examples_df : pandas.DataFrame or None, optional
        DataFrame containing auxiliary examples and/or keywords associated
        with codes.
    n_examples : int, default=10
        Maximum number of examples to sample per code when building the
        dictionary.

    Attributes
    ----------
    notes_df : pandas.DataFrame
        Notes data (possibly sampled).
    example_df : pandas.DataFrame or None
        Examples data.
    n_eg : int
        Maximum number of examples per code.
    notes_dict : dict
        Dictionary keyed by code. Each value is a dict that may contain
        ``'description'``, ``'example'``, and ``'key_words'``.
    logger : logging.Logger
        Instance logger created by :meth:`set_logger`.

    Returns
    -------
    None
    """
    def __init__(
            self,
            notes_df=None,
            sample=None,
            random_state=23,
            examples_df=None,
            n_examples=10
            ):
        self.random_state = random_state
        if not sample:
            self.notes_df = notes_df
        else:
            self.notes_df = notes_df.sample(n=sample, random_state=random_state)
        self.example_df = examples_df
        self.n_eg = n_examples
        self.notes_dict = {}
        self.set_logger()

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
    def preprocess_text(text):
        """
        Preprocess a text string.

        This base implementation is a no-op that returns the input unchanged.
        Subclasses can override it to apply domain-specific normalization.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            The (possibly) processed text. In this base class it is identical to
            the input.
        """
        return text

    def compare_keys(self, col_c, name):
        """
        Identify codes present in `notes_dict` that are missing from the examples DataFrame.

        The method compares the current keys of ``self.notes_dict`` against the
        unique values of ``self.example_df[col_c]`` and returns the missing codes.
        If any are missing, a warning is logged.

        Parameters
        ----------
        col_c : str
            Column name in ``self.example_df`` that contains code/label values.
        name : str
            Human-readable label used in the warning message (e.g., "examples",
            "key words").

        Returns
        -------
        list
            Codes that exist in ``self.notes_dict`` but do not appear in
            ``self.example_df[col_c]``.
        """
        data_c = self.notes_dict.keys()
        examples_c = self.example_df[col_c].unique()
        no_exp_key = [code for code in data_c if code not in examples_c]
        if len(no_exp_key) > 0:
            self.logger.warning(f"{no_exp_key} labels don't have {name}")
        return no_exp_key

    def set_descriptions(self, col_c: str, col_d_l: list) -> None:
        """
        Populate per-code descriptions in `notes_dict` from the notes DataFrame.

        For each row in ``self.notes_df``, this method concatenates the values of
        the columns listed in `col_d_l` (joined by newlines), applies
        :meth:`preprocess_text`, and stores the result as
        ``self.notes_dict[code]['description']`` where `code` is taken from
        ``row[col_c]``.

        Parameters
        ----------
        col_c : str
            Column name in ``self.notes_df`` containing the code/label used as the
            dictionary key.
        col_d_l : list of str
            Column names in ``self.notes_df`` whose values are concatenated to form
            the description.

        Returns
        -------
        None
            Updates ``self.notes_dict`` in-place.

        Notes
        -----
        The function signature annotates a ``str`` return type, but the current
        implementation does not return a value.
        """
        for _, row in self.notes_df.iterrows():
            desc = self.preprocess_text('\n'.join(row[name] for name in col_d_l))
            code = row[col_c]
            self.notes_dict[code] = {}
            self.notes_dict[code]['description'] = desc

    def set_examples(self, col_c: str, col_d: str):
        """
        Attach sampled example texts to each code in `notes_dict`.

        For each code in ``self.notes_dict``, this method selects matching rows in
        ``self.example_df`` (where ``example_df[col_c] == code``), samples up to
        ``self.n_eg`` values from column `col_d`, joins them with newlines, applies
        :meth:`preprocess_text`, and stores the result under ``'example'``.

        If a code is missing from the examples DataFrame, ``None`` is stored.

        Parameters
        ----------
        col_c : str
            Column name in ``self.example_df`` containing code/label values.
        col_d : str
            Column name in ``self.example_df`` containing example text values.

        Returns
        -------
        None
            Updates ``self.notes_dict`` in-place.
        """
        examples_keys = self.compare_keys(col_c, "examples")
        for code, in_dict in self.notes_dict.items():
            if code in examples_keys:
                examples = None
            else:
                eg_df = self.example_df[self.example_df[col_c] == code]
                if self.n_eg < eg_df.shape[0]:
                    n = self.n_eg
                else:
                    n = eg_df.shape[0]
                examples = self.preprocess_text('\n'.join(eg_df[col_d].sample(n=n, random_state=self.random_state).values))
            in_dict['example'] = examples

    def set_key_words(self, col_c: str, col_k: str):
        """
        Attach keyword tokens to each code in `notes_dict`.

        For each code in ``self.notes_dict``, the method selects matching rows in
        ``self.example_df`` (where ``example_df[col_c] == code``), removes
        duplicates by `col_k`, splits each keyword string on whitespace, applies
        :meth:`preprocess_text` to each token, and stores the resulting list under
        ``'key_words'``.

        If a code is missing from the examples DataFrame, ``None`` is stored.

        Parameters
        ----------
        col_c : str
            Column name in ``self.example_df`` containing code/label values.
        col_k : str
            Column name in ``self.example_df`` containing keyword strings.

        Returns
        -------
        None
            Updates ``self.notes_dict`` in-place.
        """
        examples_keys = self.compare_keys(col_c, "key words")
        for code, in_dict in self.notes_dict.items():
            if code in examples_keys:
                key_words = None
            else:
                eg_df = self.example_df[self.example_df[col_c] == code].copy()
                eg_df.drop_duplicates(subset=col_k, inplace=True)
                key_words = [self.preprocess_text(word) for value in eg_df[col_k] for word in value.split()]
                if self.n_eg < len(key_words):
                    n = self.n_eg
                else:
                    n = len(key_words)
                rng = random.Random(self.random_state)
                key_words = rng.sample(key_words, k=n)
            in_dict['key_words'] = key_words

    def get_dict_data(self, col_notes_c=None, col_notes_d_l=None, col_eg_c=None, col_eg_d=None, col_eg_k=None):
        """
        Build and return the per-code dictionary with descriptions, examples and keywords.

        Depending on which column names are provided, this method calls:
        - :meth:`set_descriptions` if `col_notes_c` and `col_notes_d_l` are provided,
        - :meth:`set_examples` if `col_eg_c` and `col_eg_d` are provided,
        - :meth:`set_key_words` if `col_eg_c` and `col_eg_k` are provided.

        Parameters
        ----------
        col_notes_c : str or None, optional
            Code column in ``self.notes_df``.
        col_notes_d_l : list of str or None, optional
            List of note columns in ``self.notes_df`` used to build descriptions.
        col_eg_c : str or None, optional
            Code column in ``self.example_df`` used to match examples/keywords.
        col_eg_d : str or None, optional
            Example text column in ``self.example_df``.
        col_eg_k : str or None, optional
            Keyword string column in ``self.example_df``.

        Returns
        -------
        dict
            The populated dictionary mapping ``code -> metadata``.
        """
        if col_notes_c and col_notes_d_l:
            self.set_descriptions(col_notes_c, col_notes_d_l)
        if col_eg_c and col_eg_d:
            self.set_examples(col_eg_c, col_eg_d)
        if col_eg_c and col_eg_k:
            self.set_key_words(col_eg_c, col_eg_k)
        return self.notes_dict


class ProcessNACENB(NotesProcess):
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess Norwegian NACE by filtering characters and normalizing spacing.

        The method removes characters not matching a restricted alphabet
        (A-Z/a-z plus ÆØÅ/æøå), selected punctuation, whitespace and newlines.
        It also fixes spacing before periods (replaces `' .'` with `'.'`).

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Cleaned text containing only allowed characters and normalized
            punctuation spacing.
        """
        pattern = r"[^A-Za-zÆØÅæøå\.\,\:\;\-\"\' \n]"
        text = re.sub(pattern, "", text)
        return text.replace(' .', '.')