# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 10:02:20 2025

@author: git.metodologia@ine.es
"""
import pandas as pd
import fasttext as ft
from time import time
import os

from codauto.utils import preprocess_text
from codauto.codifier import Codifier


class CodifierFastText(Codifier):
    """
    FastText-based text classifier extending the abstract Codifier base class.

    This class implements the abstract methods from Codifier (`train`, `save`, `load`,
    `get_pred_for_batch`) using FastText's supervised training. It supports hierarchical
    classification for CNAE codes, optional preprocessing, and training with pre-trained
    vectors.

    Inherited Attributes:
        - root_path (str): Base directory for storing/loading models and temporary files.
        - logger (logging.Logger): Logger for training and evaluation messages.
        - min_lenght_texts (int): Minimum text length for preprocessing.
        - structure: Hierarchy structure instance containing levels and reversed hierarchy.
        - model: Placeholder for the trained FastText model (initialized to None).
        - train_df (pd.DataFrame): Preprocessed training dataset.
        - test_df (pd.DataFrame): Preprocessed test dataset.
        - correspondences (pd.Series or None): Optional mapping for direct recoding of CNAE codes.

    Methods:
    --------
    load(name: str):
        Loads a pre-trained FastText model from the specified file.

    get_train_set_text(train_set: pd.DataFrame) -> list[str]:
        Converts a DataFrame of labels and texts into FastText supervised training format.
        Each line is: "__label__<label> <text>".

    train(**kwargs):
        Trains a FastText supervised model on the `train_df` or a provided dataset.

        Optional kwargs:
            - epoch (int, default=10): Number of training epochs.
            - lr (float, default=0.1): Learning rate.
            - wordNgrams (int, default=3): Max word n-gram length.
            - pretrained_vectors (str, optional): Filename of pre-trained word vectors.
            - train_set (pd.DataFrame, default=self.train_df): Training dataset.

    save(name: str):
        Saves the current FastText model to a file in `root_path`.

    get_pred_for_batch(samples: list[str], idxs: list[int], clean_samples: bool=False) -> dict:
        Predicts hierarchical labels and confidence scores for a batch of text samples.

        Parameters:
            - samples: Text samples to predict.
            - idxs: Identifiers corresponding to each sample.
            - clean_samples: If True, preprocess texts before prediction.

        Returns:
            dict mapping each index in `idxs` to a dictionary containing hierarchical
            labels ('label_<level>') and confidence scores ('conf_<level>') for each level
            in the hierarchy.

    Notes:
    ------
    - Hierarchical prediction supports aggregation from class-level predictions
      to higher levels (group, division, section).
    - Supports integration with the parent class's `get_top_n_predictions` and
      `get_preds_test_set` for batch evaluation.
    - Preprocessing leverages `preprocess_text` as used in the parent Codifier class.
    - Time taken for training is logged using the inherited `logger`.
    """

    def load(self, name):
        """
        Load a pre-trained FastText model from a file.

        This method initializes the `model` attribute by loading a FastText supervised
        model from the specified file within the `root_path`.

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

        This method formats each row of the input DataFrame as a string suitable
        for FastText training. Each line follows the pattern:
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

        This method prepares the training data in FastText format, optionally uses
        pre-trained word vectors, and trains a supervised model. Training duration
        is logged.

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
                Training dataset with labels in the first column and text in the second.

        Notes
        -----
        - The training data is temporarily written to a text file in `root_path`.
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

    def get_pred_for_batch(self, samples, idxs, clean_samples=False):
        """
        Generates hierarchical predictions for a batch of samples.

        This method takes a batch of input samples, optionally cleans
        them using a preprocessing function, obtains raw predictions
        from the model, and structures them into a hierarchical
        format according to `self.structure`.

        Parameters
        ----------
        samples : list of str
            The input texts or data samples for which predictions are generated.
        idxs : list
            Identifiers or indices corresponding to each sample in the batch.
        clean_samples : bool, optional (default=False)
            Whether to preprocess/clean the samples before prediction using
            `preprocess_text`.

        Returns
        -------
        dict
            A dictionary mapping each index in `idxs` to its hierarchical
            predictions. For each sample, the prediction dictionary contains:
            - `label_{lvl}`: List of predicted labels for hierarchy level `lvl`.
            - `conf_{lvl}`: Corresponding list of confidence scores, rounded to 4 decimals.

        Notes
        -----
        - The method assumes that `self.model` has a `predict` method returning
          raw prediction scores and labels.
        - The hierarchical structure is defined by `self.structure`, which
          should include:
            - `level_l`: List of hierarchy levels (from top to bottom).
            - `reversed_hierarchy`: Mapping from leaf labels to their hierarchy.
        - Confidence scores are summed for higher levels in the hierarchy.
        - Predictions for each level are sorted in descending order of confidence.
        - This method works with multi-level classification scenarios where
          leaf predictions propagate confidence to higher levels.

        Example
        -------
        >>> samples = ["text1", "text2"]
        >>> idxs = [0, 1]
        >>> preds = obj.get_pred_for_batch(samples, idxs, clean_samples=True)
        >>> preds[0]['label_2']  # leaf-level labels for first sample
        ['label_a', 'label_b', ...]
        >>> preds[0]['conf_2']   # corresponding confidence scores
        [0.95, 0.87, ...]
        """
        if clean_samples:
            samples = [
                preprocess_text(sample) for sample in samples
            ]

        preds_raw = self.model.predict(
            samples,
            k=len(self.model.labels)
        )

        preds_raw = zip(preds_raw[0], preds_raw[1])

        hierarchical_preds = {}
        for idx, pred_raw in zip(idxs, preds_raw):
            pred = pd.DataFrame.from_dict(
                dict(zip(pred_raw[0], pred_raw[1])),
                orient='index'
            )
            whole_pred = pd.DataFrame.from_dict(
                pred.apply(
                    lambda row: self.structure.reversed_hierarchy[
                        row.name.replace('__label__', '')
                    ], axis=1
                ).to_dict(),
                orient='index'
            ).merge(
                pred, left_index=True, right_index=True
            ).rename(
                columns={0: 'conf'}
            ).set_index(
                self.structure.level_l[-1]
            ).sort_values(
                'conf', ascending=False
            )

            hierarchical_pred = dict()
            for lvl in self.structure.level_l:
                hierarchical_pred[f'label_{lvl}'] = []
                hierarchical_pred[f'conf_{lvl}'] = []

            class_labels = whole_pred.index.tolist()
            class_confs = whole_pred.conf.astype(float).tolist()

            for class_label, class_conf in zip(class_labels, class_confs):
                hierarchical_pred[f'label_{self.structure.level_l[-1]}'].append(class_label)
                hierarchical_pred[f'conf_{self.structure.level_l[-1]}'].append(round(class_conf, 4))
                for lvl in self.structure.level_l[:-1]:
                    lvl_label = whole_pred.loc[class_label, lvl]
                    if lvl_label not in hierarchical_pred[f'label_{lvl}']:
                        lvl_conf = whole_pred[
                            whole_pred[lvl] == lvl_label
                        ].conf.sum().round(4)
                        hierarchical_pred[f'label_{lvl}'].append(
                            lvl_label
                        )
                        hierarchical_pred[f'conf_{lvl}'].append(
                            round(lvl_conf, 4)
                        )

            # sort for each level
            for lvl in self.structure.level_l[:-1]:
                sorted_pairs = sorted(list(zip(
                    hierarchical_pred[f'conf_{lvl}'],
                    hierarchical_pred[f'label_{lvl}']
                )), reverse=True)
                hierarchical_pred[f'label_{lvl}'] = [
                    l for c, l in sorted_pairs
                ]
                hierarchical_pred[f'conf_{lvl}'] = [
                    c for c, l in sorted_pairs
                ]

            hierarchical_preds[idx] = hierarchical_pred

        return hierarchical_preds
