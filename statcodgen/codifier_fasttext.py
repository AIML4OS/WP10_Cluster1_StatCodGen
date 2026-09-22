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
Created on Thu Aug 28 10:02:20 2025

@author: git.metodologia@ine.es
"""
import fasttext as ft
from time import time
import os

from statcodgen.codifier import Codifier


class CodifierFastText(Codifier):
    """
    FastText-based text classifier extending the abstract Codifier base class.

    This class implements the abstract methods from Codifier (`train`, `save`,
    `load`, `get_pred_for_batch`) using FastText's supervised training.

    Inherited Attributes:
        - root_path (str): Base directory for storing/loading models and
          temporary files.
        - logger (logging.Logger): Logger for training and evaluation messages.
        - min_lenght_texts (int): Minimum text length for preprocessing.
        - structure: Hierarchy structure instance containing levels and
          reversed hierarchy.
        - model: Placeholder for the trained FastText model (initialized to
          None).
        - train_df (pd.DataFrame): Preprocessed training dataset.
        - test_df (pd.DataFrame): Preprocessed test dataset.
        - correspondences (pd.Series or None): Optional mapping for direct
          recoding of CNAE codes.

    Methods:
    --------
    load(name: str):
        Loads a pre-trained FastText model from the specified file.

    get_train_set_text(train_set: pd.DataFrame) -> list[str]:
        Converts a DataFrame of labels and texts into FastText supervised
        training format.
        Each line is: "__label__<label> <text>".

    train(**kwargs):
        Trains a FastText supervised model on the `train_df` or a provided
        dataset.

        Optional kwargs:
            - epoch (int, default=10): Number of training epochs.
            - lr (float, default=0.1): Learning rate.
            - wordNgrams (int, default=3): Max word n-gram length.
            - pretrained_vectors (str, optional): Filename of pre-trained word
              vectors.
            - train_set (pd.DataFrame, default=self.train_df): Training dataset

    save(name: str):
        Saves the current FastText model to a file in `root_path`.

    get_pred_for_batch(
        samples: list[str]
    ) -> list[tuple[list[str], list[float]]]:
        Predicts labels and confidence scores for a batch of text samples.

        Parameters:
            - samples: Text samples to predict.

        Returns:
            A list where each element corresponds to one input sample and
            contains a tuple with:
                * a list of predicted labels (strings), with the '__label__'
                prefix removed,
                * a list of confidence scores (floats) for the corresponding
                labels
    """

    def load(self, name):
        """
        Load a pre-trained FastText model from a file.

        This method initializes the `model` attribute by loading a FastText
        supervised model from the specified file within the `root_path`.

        Parameters
        ----------
        name : str
            Filename of the FastText model to load (e.g., 'model.bin').

        Notes
        -----
        - The loaded model replaces any existing model stored in `self.model`.
        - The path is automatically resolved relative to `self.root_path`.
        """
        self.model = ft.load_model(os.path.join(self.root_path, name))

    @staticmethod
    def get_train_set_text(train_set):
        """
        Convert a DataFrame into FastText supervised training format.

        This method formats each row of the input DataFrame as a string
        suitable for FastText training. Each line follows the pattern:
        "__label__<label> <text>".

        Parameters
        ----------
        train_set : pandas.DataFrame
            DataFrame containing the training data. The first column should be
            the label, and the second column should be the corresponding text.

        Returns
        -------
        list of str
            A list of formatted strings ready for FastText supervised training.

        Example
        -------
        >>> df = pd.DataFrame({'label': ['A', 'B'], 'text': ['text1', 'text2']})
        >>> get_train_set_text(df)
        ['__label__A text1', '__label__B text2']
        """
        label_col = train_set.columns[0]
        text_col = train_set.columns[1]
        train_set_text = [
            f"__label__{row[label_col]} {row[text_col]}"
            for _, row in train_set.iterrows()
        ]
        return train_set_text

    def train(self, **kwargs):
        """
        Train a FastText supervised model on the provided dataset.

        This method prepares the training data in FastText format, optionally
        uses pre-trained word vectors, and trains a supervised model. Training
        duration is logged.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments:
            - epoch : int, default=10
                Number of training epochs.
            - lr : float, default=0.1
                Learning rate for training.
            - wordNgrams : int, default=3
                Maximum length of word n-grams.
            - pretrained_vectors : str or None, default=None
                Filename of pre-trained word vectors to use.
            - train_set : pandas.DataFrame, default=self.train_df
                Training dataset with labels in the first column and text in
                the second.

        Notes
        -----
        - The training data is temporarily written to a text file in
          `root_path`.
        - If `pretrained_vectors` is provided, it is loaded from `root_path`.
        - The trained model replaces any existing model in `self.model`.
        - Training time is measured and logged in hours, minutes, and seconds.
        """
        start = time()
        epoch = kwargs.get('epoch', 10)
        learning_rate = kwargs.get('lr', 0.1)
        word_ngrams = kwargs.get('wordNgrams', 3)
        pretrained_vectors = kwargs.get('pretrained_vectors', None)
        train_set = kwargs.get('train_set', self.train_df)
        temp_path = os.path.join(
            self.root_path,
            'train_dataset.txt'
        )
        train_set_text = self.get_train_set_text(train_set)

        with open(temp_path, "w", encoding="utf-8") as temp_train_set:
            temp_train_set.write('\n'.join(train_set_text))

        if pretrained_vectors is not None:
            pretrained_vectors_path = os.path.join(
                self.root_path,
                pretrained_vectors
            )
            self.model = ft.train_supervised(
                input=temp_path,
                epoch=epoch, lr=learning_rate, wordNgrams=word_ngrams,
                pretrainedVectors=pretrained_vectors_path
            )
        else:
            self.model = ft.train_supervised(
                input=temp_path,
                epoch=epoch, lr=learning_rate, wordNgrams=word_ngrams,
            )
        end = time()
        total_seconds = end - start
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.logger.info("Train time: %dh %dm %ds", hours, minutes, seconds)

    def save(self, name):
        """
        Saves the trained model to a file.

        This method uses the object's `model` attribute (which is
        assumed to have a `save_model` method) and saves it to the
        path defined by `self.root_path` with the filename provided
        in the `name` parameter.

        Parameters
        ----------
        name : str
            The filename to save the model as. May include or omit
            the file extension depending on what the model's
            `save_model` method requires.

        Example
        -------
        >>> my_object.save("trained_model.bin")
        This will save the model at:
        self.root_path/trained_model.bin

        Notes
        -----
        - `self.root_path` must be a valid, existing directory.
        - The `self.model` object must implement the `save_model` method.
        - No validation of file extension is performed; it is assumed
          that the `save_model` method handles it.
        """
        self.model.save_model(os.path.join(self.root_path, name))

    def get_pred_for_batch(self, samples):
        """
        Generate predictions for a batch of input samples.

        This method calls the underlying model to obtain raw predictions
        (labels and confidence scores) for each input in the batch, and
        post-processes them by removing the `__label__` prefix from labels.

        Parameters
        ----------
        samples : list of str
            A batch of input texts to classify.

        Returns
        -------
        list of tuples
            A list where each element corresponds to one input sample.
            Each element is a tuple of two lists:
            - labels (list of str): predicted labels for the sample.
            - confidences (list of float): confidence scores for the labels.

        Notes
        -----
        This method is prepare for a fasttext model so:
            - The method assumes `self.model` provides a `predict` method that
              accepts a list of inputs and returns a tuple `(labels, scores)`,
              where:
                * `labels[i]` is the list of top-k predicted labels for sample
                i.
                * `scores[i]` is the list of corresponding confidence scores.
            - Label strings are expected to be prefixed with `__label__`, which
              this method strips out before returning.
            - The number of predictions (`k`) is set to the total number of
              labels available in the model.

        Example
        -------
        >>> samples = ["example text 1", "example text 2"]
        >>> preds = obj.get_pred_for_batch(samples)
        >>> preds[0][0]   # predicted labels for the first sample
        ['label_a', 'label_b', ...]
        >>> preds[0][1]   # corresponding confidence scores
        [0.95, 0.87, ...]
        """
        preds_raw = self.model.predict(
            samples,
            k=len(self.model.labels)
        )
        preds_zip = zip(preds_raw[0], preds_raw[1])
        clean_preds = [None] * len(samples)
        for idx, pred_zip in enumerate(preds_zip):
            label_l = [label.replace('__label__', '') for label in pred_zip[0]]
            conf_l = list(pred_zip[1])
            clean_preds[idx] = (label_l, conf_l)
        return clean_preds
