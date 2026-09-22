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
Created on Mon Jun 24 11:02:31 2024

@author: git.metodologia@ine.es
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
import logging
import warnings
from tqdm import tqdm
from abc import ABC, abstractmethod

from statcodgen.my_utils import truncate_colormap
from statcodgen.my_utils import forward_pseudolog_transform
from statcodgen.my_utils import reverse_pseudolog_transform


class Codifier(ABC):
    """
    Abstract base class for hierarchical codifiers.

    This class handles preprocessing, cleaning, and validation of
    training and testing datasets for hierarchical classification tasks.
    It also manages correspondences between old and new classification
    codes and provides methods for predictions, evaluation, and metrics.

    Attributes
    ----------
    root_path : str
        Path to store models and temporary files.
    logger : logging.Logger
        Logger configured for the codifier instance.
    min_lenght_texts : int
        Minimum number of characters required for a description to be valid.
    structure : object
        An object defining the hierarchical structure of codes.
        Must expose `level_l` (list of levels) and `reversed_hierarchy`
        (mapping from leaf codes to hierarchical paths).
    model : object or None
        Trained model instance. Initialized as None.
    correspondences : dict or None
        Mapping of old codes to new codes across hierarchy levels,
        if provided via `corres_df`.
    train_df : pandas.DataFrame or None
        Cleaned and validated training dataset.
    test_df : pandas.DataFrame or None
        Cleaned and validated testing dataset with hierarchical labels.
    preprocess : callable function or None 
        If a text preprocessing function is given, applies such function.
        If None, does not apply text preprocessing. 
        Default Value: None.         

    Notes
    -----
    - This is an abstract base class. Subclasses must implement
      `train`, `save`, `load`, and `get_pred_for_batch`.
    """

    def __init__(
            self,
            structure_instance,
            root_path,
            train_df=None,
            test_df=None,
            corres_df=None,
            min_lenght_texts=3,
            preprocess = None
    ):
        self.root_path = root_path
        os.makedirs(self.root_path, exist_ok=True)
        self.logger = logging.getLogger(f'Codifier.{id(self)}')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(stream_handler)
        self.logger.propagate = True
        self.min_lenght_texts = min_lenght_texts
        self.structure = structure_instance
        self.model = None
        self.preprocess = preprocess
        self.correspondences = None
        self.train_df = None
        self.test_df = None
        if corres_df is not None:
            self.correspondences = self.get_correspondences(corres_df)
        if train_df is not None:
            self.train_df = self.get_train_dataset(train_df, 'train_df')
        if test_df is not None:
            self.test_df = self.get_test_dataset(test_df, 'test_df')
    
    def get_correspondences(self, correspondences_df):
        """
        Convert a correspondence DataFrame into a hierarchical mapping.

        Parameters
        ----------
        correspondences_df : pandas.DataFrame
            A DataFrame with 3 columns:
            [0] old_code : str
                Previous classification code.
            [1] new_code : str
                Target classification code.
            [2] level : int
                Hierarchical level of the correspondence. It must be the new
                code's (new_code) level.'

        Returns
        -------
        pandas.Series
            A Series indexed by hierarchy level. Each element is a mapping
            from old codes to sets of new codes at that level.

        Raises
        ------
        ValueError
            If the DataFrame does not have exactly 3 columns.

        Notes
        -----
        - Periods in codes are removed before processing.
        - Useful for direct recoding of legacy classification systems.
        """
        if correspondences_df.shape[1] != 3:
            raise ValueError(
                "corres_df must have 3 columns: [0] previous classification code, [1] new classification code, [2]  hierarchical level of the correspondence"
            )
        correspondences_df.columns = ['old_code', 'new_code', 'level']
        correspondences_df = correspondences_df.map(
            lambda v: v.replace('.', '')
        )
        correspondences_df['level'] = correspondences_df['level'].apply(int)
        correspondences = correspondences_df.groupby(['level', 'old_code'])['new_code'].apply(set)
        return correspondences

    def get_code(self, code):
        """
        Validate and standardize a classification code.

        Parameters
        ----------
        code : str
            Input code to check against the hierarchy.

        Returns
        -------
        str or float
            - The same code if it exists in `self.structure.reversed_hierarchy`
            - np.nan if the code is not found.

        Notes
        -----
        Ensures only valid codes are retained before predictions or evaluation.
        """
        if code in self.structure.reversed_hierarchy.keys():
            return code
        return np.nan

    def clean_data(self, data_df):
        """
        Clean and preprocess a dataset by normalizing codes and descriptions.

        Parameters
        ----------
        data_df : pandas.DataFrame
            Dataset with at least two columns:
            - Column 0: code
            - Column 1: description

        Returns
        -------
        pandas.DataFrame
            Cleaned dataset where:
            - Codes are standardized and validated.
            - Descriptions are preprocessed when a preprocessing function is provided.
            - Descriptions shorter than `min_lenght_texts` are replaced with NaN.

        Processing Steps
        ----------------
        1. Extracts the first two columns of the dataframe.
        2. Standardizes the codes in the first column using droping dots and
        get_code.
        3. Applies optional text preprocessing to clean the text in the second
        column.
        4. Replaces text entries that are shorter than `min_lenght_texts`
           with NaN.
        """
        col_l = data_df.columns
        col_n_0 = col_l[0]
        col_n_1 = col_l[1]
        data_df[col_n_0] = data_df[col_n_0].apply(
            lambda n: self.get_code(str(n).replace('.', ''))
        )
        if self.preprocess is not None:
            data_df[col_n_1] = data_df[col_n_1].apply(self.preprocess)
        
        data_df[col_n_1] = data_df[col_n_1].apply(
                lambda desc: np.nan if len(desc) <= self.min_lenght_texts else desc
        )
        return data_df

    def load_data(self, data_df, data_name='data_df'):
        """
        Load, clean, and validate a dataset.

        Parameters
        ----------
        data_df : pandas.DataFrame
            Raw dataset to load.
        data_name : str, optional
            Name used in log messages (default is 'data_df').

        Returns
        -------
        pandas.DataFrame
            Cleaned dataset with invalid codes and short descriptions removed,
            and index reset.

        Process Overview
        ----------------
        1. Logs the start of the loading process.
        2. Copies the input dataframe to avoid modifying the original.
        3. Logs the raw number of entries in the dataset.
        4. Cleans the dataframe using the `clean_data` method:
           - Standardizes codes.
           - Preprocesses text descriptions.
           - Replaces too-short descriptions with NaN.
        5. Counts and logs the number of invalid codes and descriptions
        (NaN values).
        6. Drops all rows with NaN values.
        7. Resets the dataframe index.
        8. Logs the number of entries after pruning.
        9. Returns the cleaned dataframe.
        """
        self.logger.info(f'Loading {data_name}..')
        n_data = len(data_df)
        input_df = data_df.copy()
        self.logger.info(f'{data_name} raw data count: {n_data}')
        out_df = self.clean_data(input_df)
        bad_codes = out_df.iloc[:, 0].isna().sum()
        bad_desc = out_df.iloc[:, 1].isna().sum()
        self.logger.info(f'{data_name} bad codes count: {bad_codes}')
        self.logger.info(f'{data_name} bad descriptions count: {bad_desc}')
        out_df.dropna(inplace=True)
        out_df.reset_index(drop=True, inplace=True)
        n_data_pruned = len(out_df)
        self.logger.info(f'{data_name} pruned data count: {n_data_pruned}')
        return out_df

    def check_codes(self, data_df, name, show_codes=True):
        """
        Check which hierarchical codes are missing in a dataset.

        Parameters
        ----------
        data_df : pandas.DataFrame
            Dataset to check. First column is assumed to contain codes.
        name : str
            Name used in logging.
        show_codes : bool, optional
            If True, logs the list of missing codes (default=True).

        Returns
        -------
        int
            Number of codes defined in the hierarchy that are not
            represented in the dataset.

        Process Overview
        ----------------
        1. Extracts the unique codes present in the dataset.
        2. Retrieves all codes defined in the hierarchical structure.
        3. Computes the set difference to find unrepresented codes.
        4. Logs the number of missing codes and optionally the list of these
        codes.
        5. Returns the count of unrepresented codes.
        """
        codes_set = set(data_df.iloc[:, 0].unique())
        all_codes = set(self.structure.reversed_hierarchy.keys())
        unrepresented_codes = all_codes - codes_set
        len_unre_codes = len(unrepresented_codes)
        if show_codes:
            text = f'{name} has {len_unre_codes} codes unrepresented: {unrepresented_codes}'
        else:
            text = f'{name} has {len_unre_codes} codes unrepresented'
        if len_unre_codes != 0:
            self.logger.info(text)
        return len_unre_codes

    def get_train_dataset(self, train_dataset, name):
        """
        Clean and validate the training dataset.

        Parameters
        ----------
        train_dataset : pandas.DataFrame
            Raw training dataset with at least 2 columns:
            [0] code, [1] description.
        name : str
            Name used in logging.

        Returns
        -------
        pandas.DataFrame
            Cleaned training dataset with only valid codes and descriptions.

        Process Overview
        ----------------
        1. Checks that the dataset has at least 2 columns. Raises an error
        otherwise.
        2. Warns if the dataset has more than 2 columns, but only uses the
        first two.
        3. Cleans the dataset using `load_data`
        (removes invalid codes/descriptions).
        4. Checks for unrepresented codes using `check_codes`.
        5. Issues a warning if some codes in the hierarchy are missing from the
        dataset.
        6. Returns the cleaned dataset ready for training.
        """

        if train_dataset.shape[1] < 2:
            raise ValueError(
                "train_df must have at least 2 columns: [0] code, [1] description"
            )
        if train_dataset.shape[1] > 2:
            warnings.warn(
                "train_dataset has more than 2 columns. Only columns 0 (expected: code) and 1 (expected: description) will be used."
            )
        out_df = self.load_data(train_dataset, name)
        len_unre_codes = self.check_codes(out_df, name)
        if len_unre_codes != 0:
            warnings.warn(
                f"{name} has {len_unre_codes} codes unrepresented, look log doc"
            )
        return out_df

    def get_test_dataset(self, test_dataset, name):
        """
        Clean and validate the test dataset, and generate hierarchical ground
        truth.

        Parameters
        ----------
        test_dataset : pandas.DataFrame
            Raw test dataset with 2 obligatory columns and an aditional optional one containing the source:
            [0] code, [1] description, (optional:) [2] source.
        name : str
            Name used in logging.

        Returns
        -------
        pandas.DataFrame
            Processed dataset with columns:
            - 'desc': preprocessed description text.
            - 'source': sample source (when source is provided).
            - 'gt_<level>': ground-truth codes for each hierarchical level.

        Method Details
        --------------
        1. Validates that the input DataFrame has 2 or 3 columns; raises
           ValueError if not.
        2. Cleans the dataset using `load_data` to handle invalid or too short
           codes/descriptions.
        3. Renames columns to standard format:
           - Top-level code: 'gt_<highest_level>'
           - Description: 'desc'
           - Source: 'source'
        4. Expands the top-level code into ground-truth codes for all
           hierarchical levels using `self.structure.reversed_hierarchy`.
        5. Returns a DataFrame ready for hierarchical evaluation.
        """
        if test_dataset.shape[1] not in [2, 3]:
            raise ValueError(
                "test_df must have 2 or 3 columns: [0] code, [1] description, (optional:) [2] source"
            )
        input_df = self.load_data(test_dataset, name)
        if input_df.shape[1] == 3:
            input_df.columns = [
            f'gt_{self.structure.level_l[-1]}', 'desc', 'source'
            ]
        else:
            input_df.columns = [
            f'gt_{self.structure.level_l[-1]}', 'desc'
            ]
        ground_truth = f'gt_{self.structure.level_l[-1]}'
        for lvl in self.structure.level_l:
            input_df[f'gt_{lvl}'] = input_df[ground_truth].apply(
                lambda c: self.structure.reversed_hierarchy[c][lvl]
                if c in self.structure.reversed_hierarchy else None
            )

        return input_df

    @abstractmethod
    def train(self, **kwargs):
        """
        Abstract method to train the model.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for training configuration.

        Notes
        -----
        This method must be implemented in any subclass of Codifier. It should
        handle the actual training procedure of the underlying model, storing
        the trained model in `self.model`.

        Native filtering requires the training structure to include
        garbage classes (``ensure_garbage_class=True``).
        """

    @abstractmethod
    def save(self, name):
        """
        Abstract method to save the trained model to disk.

        Parameters
        ----------
        name : str
            The filename (or path) under which the trained model will be saved.

        Notes
        -----
        This method must be implemented in any subclass of Codifier. It should
        handle serialization of the trained model (`self.model`) so that it can
        be reloaded later using the `load` method.
        """

    @abstractmethod
    def load(self, name):
        """
        Abstract method to load a previously saved model from disk.

        Parameters
        ----------
        name : str
            The filename (or path) from which the trained model should be
            loaded.

        Notes
        -----
        This method must be implemented in any subclass of Codifier. It should
        handle deserialization of the model and assign it to `self.model` so
        that the codifier can be used for predictions.
        """

    @abstractmethod
    def get_pred_for_batch(self, samples):
        """
        Generate predictions for a batch of input samples.

        Parameters
        ----------
        samples : list of str
            Text descriptions to classify.

        Returns
        -------
        list of tuple
            Each element corresponds to one input sample and contains:
            - labels (list of str): predicted labels.
            - confidences (list of float): associated confidence scores.

        Notes
        -----
        Must be implemented by leaves classes.
        The output format is later transformed into hierarchical predictions
        by `take_preds_bottom_up`.
        """

    def take_preds_bottom_up(self, samples, idxs, clean_samples=False):
        """
        Generate hierarchical predictions from leaf-level model outputs.

        This method takes raw predictions (labels and confidences) from
        `get_pred_for_batch` and aggregates them bottom-up through the
        hierarchy defined in `self.structure`. Confidence scores from
        leaf labels are propagated and summed to compute scores at
        higher levels.

        Parameters
        ----------
        samples : list of str
            Input text samples to classify.
        idxs : list of int
            Identifiers corresponding to each input sample.
        clean_samples : bool, optional (default=False)
            If True, applies text preprocessing to the samples before
            prediction.

        Returns
        -------
        dict
            A dictionary mapping each element in `idxs` to its hierarchical
            predictions. For each sample, the value is a dictionary with:
            - 'label_<level>' : list of str
                Predicted labels at hierarchy level `<level>`, ordered
                by confidence (descending).
            - 'conf_<level>' : list of float
                Confidence scores for the corresponding labels.

        Notes
        -----
        - Predictions at the deepest level (`level_l[-1]`) are taken
          directly from `get_pred_for_batch`.
        - For higher levels, scores are aggregated by summing the
          confidences of their descendant labels.
        - If the hierarchy has only one level, only leaf predictions
          are returned.
        - The output is later consumed by methods such as
          `get_top_n_predictions` and `predict`.

        Example
        -------
         >>> samples = ["The new smartphone has an excellent camera.",
         ...            "I love hiking in the mountains."]
         >>> idxs = [101, 102]
         >>> preds = obj.take_preds_bottom_up(samples, idxs)
         >>> preds[101]['label_level_2']
         ['Electronics', 'Other']
         >>> preds[101]['conf_level_2']
         [0.85, 0.15]
         >>> preds[101]['label_level_1']
         ['Products', 'Misc']
         >>> preds[101]['conf_level_1']
         [0.85, 0.15]
        """
        pred_dict = {}
        if clean_samples:
            if self.preprocess is not None:
                samples = [
                    self.preprocess(sample) for sample in samples
                ]
        preds_confs = self.get_pred_for_batch(
            samples=samples
        )
        for pred_conf, idx in zip(preds_confs, idxs):
            hierarchical_pred = {}
            labels = pred_conf[0]
            conf = pred_conf[1]
            hierarchical_pred[f'label_{self.structure.level_l[-1]}'] = labels
            hierarchical_pred[f'conf_{self.structure.level_l[-1]}'] = conf
            if len(self.structure.level_l) == 1:
                pred_dict[idx] = hierarchical_pred
                next
            for lvl in self.structure.level_l[:-1]:
                lvl_labels = []
                lvl_probs = []
                parent_map = {}
                for lbl, prob in zip(labels, conf):
                    parent_lbl = self.structure.reversed_hierarchy[lbl][lvl]
                    parent_map.setdefault(parent_lbl, 0)
                    parent_map[parent_lbl] += prob
                sorted_pairs = sorted(
                    parent_map.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for parent_lbl, total_prob in sorted_pairs:
                    lvl_labels.append(parent_lbl)
                    lvl_probs.append(total_prob)
                hierarchical_pred[f'label_{lvl}'] = lvl_labels
                hierarchical_pred[f'conf_{lvl}'] = lvl_probs
            pred_dict[idx] = hierarchical_pred
        return pred_dict

    def get_top_n_predictions(
            self, samples, idxs, n_classes=5, clean_samples=False
    ):
        """
        Generate the top-N predictions for a batch of input samples.

        Parameters
        ----------
        samples : list of str
            A list of text descriptions to classify.
        idxs : list of int
            A list of indices corresponding to each sample in the batch.
        n_classes : int, optional (default=5)
            The number of top predictions to return per hierarchical level.
        clean_samples : bool, optional (default=False)
            Whether to apply preprocessing/cleaning to the samples before
            prediction.

        Returns
        -------
        dict
            A dictionary where each key is an index from `idxs`, and each value
            is another dictionary containing the top-N predicted labels and
            their corresponding confidence scores for each hierarchical level.
            Keys follow the format:
            - 'label_<level>': list of top-N predicted labels
            - 'conf_<level>': list of corresponding confidence scores

        Notes
        -----
        This method calls `take_preds_bottom_up` to obtain raw predictions,
        then filters them to only keep the top-N predictions per level. It is
        useful for scenarios where only the most likely predictions are needed,
        such as displaying suggestions or computing metrics.
        """
        hierarchical_preds = self.take_preds_bottom_up(
            samples, idxs, clean_samples
        )
        top_n_predictions = {}
        for idx, pred in hierarchical_preds.items():
            top_n_pred = {}
            for lvl in pred.keys():
                if 'label_' in lvl:
                    conf_key = lvl.replace('label_', 'conf_')
                    top_n_pred[lvl] = pred[lvl][:n_classes]
                    top_n_pred[conf_key] = pred[conf_key][:n_classes]
            top_n_predictions[idx] = top_n_pred
        return top_n_predictions

    def get_preds_test_set(self, test_set, n_classes):
        """
         Generate predictions for an entire test dataset using the top-N
         prediction strategy.

         Parameters
         ----------
         test_set : pandas.DataFrame
             A DataFrame containing the test samples. Must include a column
             'desc' with text descriptions.
         n_classes : int
             The number of top predictions to return for each sample and
             hierarchical level.

         Returns
         -------
         pandas.DataFrame
             A DataFrame where each row corresponds to a sample from
             `test_set`, indexed by the original DataFrame index. Each column
             contains the top-N predicted labels and confidence scores per
             hierarchical level.

         Notes
         -----
         - Iterates over each row of the test dataset.
         - Calls `get_top_n_predictions` for each description individually.
         - Collects predictions in a dictionary and converts it to a DataFrame
           for easier analysis and merging with ground truth labels.
         - Useful for evaluating the model across the full test dataset or
           generating prediction reports.
         """
        preds_test = dict()
        for row_i, row in tqdm(
            test_set.iterrows(),
            total=len(test_set),
            desc=f"Predicting test_set for {n_classes} classes"
        ):
            sample_pred = self.get_top_n_predictions(
                [row['desc']], [row_i], n_classes
            )
            preds_test[row_i] = sample_pred[row_i]
        return pd.DataFrame.from_dict(preds_test, orient='index')

    def get_precision_vs_recall_multi_conf(self, df_gt, n_classes):
        """
        Compute precision, recall, and average number of predicted classes
        across multiple confidence thresholds for each hierarchical level.

        Parameters
        ----------
        df_gt : pandas.DataFrame
            Test dataset containing, per hierarchical level, columns
            'label_{lvl}' (predicted labels ordered by confidence),
            'conf_{lvl}' (their associated confidence scores), and
            'gt_{lvl}' (ground truth label).
        n_classes : int
            Maximum number of top predictions to consider per level
            (capped automatically if a given level has fewer available
            classes; this cap is computed independently per level).

        Returns
        -------
        dict
            Dictionary keyed by hierarchical level, each value a DataFrame
            indexed by confidence threshold with columns:
                - 'precision': Fraction of predictions above the threshold
                  that are correct.
                - 'recall': Fraction of samples with at least one prediction
                  above the threshold (coverage).
                - 'mean_classes': Average number of predicted classes above
                  the threshold.

        Notes
        -----
        - Iterates through confidence thresholds from 0 to 1 with step 0.0025.
        - For each level, precomputes the confidence associated with the
          ground truth label (if present in the top-n_classes predictions)
          once, then reuses it across all thresholds for efficiency.
        - Useful for generating precision-recall curves and analyzing model
          confidence behavior across hierarchical levels.
        """
        prec_recall_mean_classes_d = dict()
        df_gt_n_classes = df_gt.copy()
        for lvl in self.structure.level_l:
            n_labels = len(df_gt[f'label_{lvl}'].iloc[0])
            n_classes_lvl = min(n_classes, n_labels)
            df_gt_n_classes[f'label_{lvl}'] = df_gt[f'label_{lvl}'].apply(
                lambda l: l[:n_classes_lvl]
            )
            df_gt_n_classes[f'conf_{lvl}'] = df_gt[f'conf_{lvl}'].apply(
                lambda l: l[:n_classes_lvl]
            )
            def _conf_del_gt(row, lvl=lvl):
                gt = str(row[f'gt_{lvl}'])
                labels = row[f'label_{lvl}']
                if gt in labels:
                    return row[f'conf_{lvl}'][labels.index(gt)]
                return np.nan
            df_gt_n_classes[f'conf_gt_{lvl}'] = df_gt_n_classes.apply(
                lambda row: _conf_del_gt(row), axis=1
            )
            conf_lists = df_gt_n_classes[f'conf_{lvl}'].apply(np.array)
            prec_recall_mean_classes = dict()
            n_total = len(df_gt_n_classes)
            for th in np.arange(0, 1.001, 0.0025):
                n_classes_th = conf_lists.apply(lambda arr: (arr > th).sum())
                n_classes_mean = n_classes_th[n_classes_th > 0.1].mean()
                covered_mask = conf_lists.apply(lambda arr: (arr > th).any())
                hit_mask = df_gt_n_classes[f'conf_gt_{lvl}'] > th
                n_samples = covered_mask.sum()
                n_correct_samples = (covered_mask & hit_mask).sum()
                recall = n_samples / n_total
                precision = n_correct_samples / n_samples if n_samples > 0 else 0
                prec_recall_mean_classes[th] = {
                    'precision': precision,
                    'recall': recall,
                    'mean_classes': n_classes_mean
                }
            prec_recall_mean_classes_d[lvl] = pd.DataFrame.from_dict(
                prec_recall_mean_classes, orient='index'
            ).dropna()
        return prec_recall_mean_classes_d

    def plot_evaluate_curve(self, df_gt, source='all', version='09'):
        """
        Plot precision vs recall curves for different numbers of predicted
        classes (1 and 15) across all hierarchical levels, highlighting
        specific confidence thresholds.

        Parameters
        ----------
        test_set : pandas.DataFrame
            The test dataset containing ground truth labels, descriptions, and
            source.
        source : str, default 'all'
            If not 'all', filters the test set to include only rows from this
            source.
        version : str, default '09'
            A string identifier for the model version or evaluation run, used
            in plot titles.

        Notes
        -----
        - Computes precision and recall using
          `get_precision_vs_recall_multi_conf` for n_classes=1 & n_classes=15.
        - Highlights four confidence thresholds: 0.8, 0.25, 0.1, and 0.03 on
          the curves.
        - Uses color and marker size to visualize the average number of
          predicted output codes.
        - Applies a pseudologarithmic x-scale to improve visualization for
          high-recall regions.
        - Displays a separate plot for each hierarchical level in the structure
        - Provides logging when levels or thresholds are missing in the
          computed precision-recall data.

        Visualization
        -------------
        - Scatter plot for n_classes=15 uses a color map representing the
          average number of output codes.
        - Scatter plot for n_classes=1 uses black markers.
        - Blue '+' markers indicate selected confidence thresholds on the
          curves.
        """
        precision_vs_recall_multi_conf_15 = self.get_precision_vs_recall_multi_conf(
            df_gt, n_classes=15
        )
        precision_vs_recall_multi_conf_1 = self.get_precision_vs_recall_multi_conf(
            df_gt, n_classes=1
        )
        THRESHOLDS = (0.8, 0.25, 0.1, 0.03)
        cmap = plt.get_cmap('gist_stern')
        new_cmap = truncate_colormap(cmap, 0, 0.8)

        for lvl in self.structure.level_l:
            if (
                lvl not in precision_vs_recall_multi_conf_15
            ) or (
                lvl not in precision_vs_recall_multi_conf_1
            ):
                self.logger.info(
                    f"Level {lvl} not found in precision_vs_recall data."
                )
                continue
            fig, ax = plt.subplots(figsize=(15, 6))
            for precision_vs_recall, n in (
                    (precision_vs_recall_multi_conf_15, 15),
                    (precision_vs_recall_multi_conf_1, 1)
            ):
                if n == 15:
                    scatter = ax.scatter(
                        precision_vs_recall[lvl]['recall'],
                        precision_vs_recall[lvl]['precision'],
                        c=precision_vs_recall[lvl]['mean_classes'],
                        cmap=new_cmap,
                        s=3 *
                        precision_vs_recall[lvl]['mean_classes'] ** 2,
                        marker='o'
                    )
                    fig.colorbar(scatter, label='Avg. Num. Output Codes')
                else:
                    scatter = ax.scatter(
                        precision_vs_recall[lvl]['recall'],
                        precision_vs_recall[lvl]['precision'],
                        c='black',
                        s=3 *
                        precision_vs_recall[lvl]['mean_classes'] ** 2,
                        marker='o'
                    )
                for th in THRESHOLDS:
                    if th in precision_vs_recall[lvl].index:
                        ax.scatter(
                            precision_vs_recall[lvl].loc[th, 'recall'],
                            precision_vs_recall[lvl].loc[th, 'precision'],
                            color='blue',
                            marker='+',
                            s=80
                        )
                        ax.text(
                            precision_vs_recall[lvl].loc[
                                th, 'recall'
                            ] + 0.0005,
                            precision_vs_recall[lvl].loc[
                                th, 'precision'
                            ] + 0.003,
                            f'{th}',
                            size=9,
                            color='blue'
                        )
                    else:
                        self.logger.info(
                            f"Threshold {th} not found for level {lvl} in precision_vs_recall_multi_conf_{n}."
                        )
            ax.set_xlim((0, 1.002))
            all_precisions = pd.concat([
                precision_vs_recall_multi_conf_15[lvl]['precision'],
                precision_vs_recall_multi_conf_1[lvl]['precision']
            ])
            ymin = max(0, all_precisions.min() - 0.02)
            ymax = min(1.01, all_precisions.max() + 0.02)
            ax.set_ylim((ymin, ymax))
            ax.set_xscale('function', functions=(
                forward_pseudolog_transform,
                reverse_pseudolog_transform
            ))
            ax.set_xticks([0.2, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 0.997, 1])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(
                f'Precision vs Recall for {version} {lvl}', fontsize=18
            )
            ax.grid(True)
            plt.show()

    def get_top_k_accuracy(self, df_pred, k):
        df_pred.columns = ["gt", "labels"]
        n_labels = len(df_pred["labels"].iloc[0])
        if k > n_labels:
            k=n_labels
        hits = df_pred.apply(
            lambda row: row["gt"] in row["labels"][:k],
            axis=1
        )
        return hits.mean()


    def evaluate(self,
                 source='all',
                 get_curve=True,
                 simplify=True,
                 version='model',
                 k=5
                 ):
        """
        Evaluate the model's performance on the test dataset by computing
        metrics and optionally plotting curves.

        Parameters
        ----------
        source : str, default 'all'
            If not 'all', filters the test set to include only rows from this
            source.
        get_curve : bool, default True
            If True, plots precision vs recall curves using
            `plot_evaluate_curve`.
        simplify : bool, default True
            If True, returns only a dictionary of metrics (accuracy, F1 scores)
            If False, also returns a detailed classification report for each
            hierarchical level.
        version : str, default 'model'
            Identifier for the evaluation run, used in plot titles.

        Returns
        -------
        dict
            If `simplify=True`, returns a dictionary containing accuracy and F1
            scores (micro, macro, weighted) per hierarchical level.
        tuple
            If `simplify=False`, returns a tuple `(metrics, report)` where
            `report` contains the detailed classification reports per level.

        Notes
        -----
        - Filters the test set based on the provided source.
        - Checks for missing or unrepresented codes in the test set using
          `check_codes`.
        - Generates predictions for the test set with `get_preds_test_set`.
        - Computes standard evaluation metrics for each hierarchical level.
        - If `get_curve` is True, calls `plot_evaluate_curve` to visualize
          precision vs recall for n_classes=1 and 15.
        - Uses zero_division=0 for F1 scores to handle cases where a class has
          no predicted samples.
        """
        # verifies if test_df is setted
        if self.test_df is None:
            raise ValueError('Cannot evaluate: parameter "test_df" must not be None. Assign a test DataFrame to the instance before calling evaluate().')
        # choose witch source from 3th column from test set is used for evaluation
        if source != 'all':
            if 'source' not in self.test_df.columns:
                raise ValueError('Source code filtering requested but no source column present in "test_df".')
            if source not in self.test_df['source'].values:
                raise ValueError(
                    f'The source value {source} is not present in the source column.'
                )
            test_set = self.test_df[self.test_df['source'] == source]
        else:
            test_set = self.test_df.copy()
        self.logger.info(
            f'Test set size {len(test_set)} for source {source}'
        )
        # clean codes test set
        self.check_codes(test_set, f'test_set_{source}', False)
        # Se ejecuta get_preds_test_set
        preds_test_set = self.get_preds_test_set(
            test_set=test_set, n_classes=len(self.structure.reversed_hierarchy)
        )
        df_gt = test_set.merge(
            preds_test_set, left_index=True, right_index=True
        )
        metrics = dict()
        report = dict()
        if get_curve:
            self.plot_evaluate_curve(
                df_gt=df_gt,
                version=version
            )
        for lvl in self.structure.level_l:
            df_gt_list = df_gt[f'gt_{lvl}'].astype(str).tolist()
            pred_label = [label[0] for label in df_gt[f'label_{lvl}']]
            accuracy = accuracy_score(df_gt_list, pred_label)
            top_k_accuracy = self.get_top_k_accuracy(df_pred=df_gt[[f'gt_{lvl}', f'label_{lvl}']], k=k)
            f1_micro = f1_score(
                df_gt_list, pred_label,
                average='micro', zero_division=0
            )
            f1_macro = f1_score(
                df_gt_list, pred_label,
                average='macro', zero_division=0
            )
            f1_weighted = f1_score(
                df_gt_list, pred_label,
                average='weighted', zero_division=0
            )
            metrics[lvl] = {
                'accuracy': accuracy,
                f'top_{k}_accuracy': top_k_accuracy,
                'f1_score': {
                    'micro': f1_micro,
                    'macro': f1_macro,
                    'weighted': f1_weighted
                }
            }
            if not simplify:
                report[lvl] = classification_report(
                    df_gt_list,
                    pred_label,
                    labels=sorted(list(set(df_gt_list)))
                )

        if simplify:
            return metrics
        else:
            return metrics, report

    def predict(
        self, desc_l, mode, hierarchical_level, threshold,
        original_code_l=None, identifier_l=None
    ):
        """
        Predict hierarchical classification codes for a list of descriptions.

        Parameters
        ----------
        desc_l : list of str
            List of textual descriptions to classify.
        mode : str
            Mode of prediction:
            - "codification": returns the top 1 predicted class.
            - "assistance": returns the top 15 predicted classes.
        hierarchical_level : int
            Target hierarchical level for which the predictions are returned.
        threshold : float
            Minimum confidence value (0-1) to include a predicted label.
        original_code_l : list, optional
            Original codes for direct recoding. If provided, predictions may be
            overridden by direct correspondences for codes that map
            unambiguously.
        identifier_l : list, optional
            Unique identifiers corresponding to each description. Defaults to
            sequential integers.

        Returns
        -------
        list of dict
            Each element corresponds to a description and contains:
            - 'description': input text.
            - 'original_code': original code if provided.
            - 'label': predicted label(s) for the specified hierarchical level.
            - 'confidence': confidence score(s) as percentages.
            - 'hierarchical_level': the level of prediction.
            - 'identifier': the identifier corresponding to the description.
            - 'title': title(s) corresponding to the predicted label(s).

        Notes
        -----
        - Handles direct recoding using `self.correspondences` if
          `original_code_l` is provided.
        - For non-direct recoding, predictions are obtained from
          `take_preds_bottom_up`.
        - Confidence scores are normalized if original codes restrict possible
          labels.
        - Predictions are filtered based on the confidence threshold and
          limited to `n_classes` according to mode.
        """
        if identifier_l is None:
            identifier_l = list(range(len(desc_l)))
        if mode == "codification":
            n_classes = 1
        elif mode == "assistance":
            n_classes = 15
        else:
            raise ValueError

        hierarchical_level_name = self.structure.level_l[
            hierarchical_level - 1
        ]

        idxs = range(len(desc_l))
        desc_l_not_direct_recoding = desc_l
        idxs_direct_recoding = []
        original_code_l_not_direct_recoding = (
            original_code_l
            if isinstance(original_code_l, list) else [None] * len(desc_l)
        )
        prediction_l = [None] * len(desc_l)

        if original_code_l is not None and any(
            oc is not None for oc in original_code_l
        ):
            if self.correspondences is None:
                raise ValueError(
                    'If original_code_l is informed, the correspondences parameter must be informed too'
                )
            for idx, original_code in zip(idxs, original_code_l):
                if original_code is not None:
                    filt = self.correspondences[hierarchical_level][
                        original_code
                    ]
                    if len(filt) == 1:  # direct recoding
                        idxs_direct_recoding.append(idx)
                        prediction_l[idx] = {
                            'description': desc_l[idx],
                            'original_code': original_code_l[idx],
                            'label': tuple(filt),
                            'confidence': (100,),
                            'hierarchical_level': hierarchical_level,
                            'identifier': identifier_l[idx],
                            'title': (self.structure.titles[tuple(filt)[0]],)
                        }
            idxs_not_direct_recoding = [
                idx for idx in idxs
                if idx not in idxs_direct_recoding
            ]
            desc_l_not_direct_recoding = [
                desc
                for idx, desc in enumerate(desc_l)
                if idx in idxs_not_direct_recoding
            ]
            original_code_l_not_direct_recoding = [
                orig_code
                for idx, orig_code in enumerate(original_code_l)
                if idx in idxs_not_direct_recoding
            ]
        else:
            idxs_not_direct_recoding = [
                idx
                for idx, _ in enumerate(idxs)
            ]
            desc_l_not_direct_recoding = [
                desc
                for idx, desc in enumerate(desc_l)
                if idx in idxs_not_direct_recoding
            ]

            original_code_l = [None]*len(desc_l)

        raw_predictions = self.take_preds_bottom_up(
            desc_l_not_direct_recoding,
            idxs_not_direct_recoding,
            callable(self.preprocess)
        )

        for idx, original_code in zip(
            idxs_not_direct_recoding,
            original_code_l_not_direct_recoding
        ):
            pred_df = pd.DataFrame({
                'labels': raw_predictions[idx][
                    f'label_{hierarchical_level_name}'
                ],
                'raw_confs': raw_predictions[idx][
                    f'conf_{hierarchical_level_name}'
                ],
                'confs': None,
            })
            if original_code is None:
                pred_df_filtered = pred_df.copy()
                pred_df_filtered['confs'] = pred_df['raw_confs']
            else:
                filt = [
                    cat in self.correspondences[
                        hierarchical_level
                    ][
                        original_code
                    ] for cat in pred_df['labels']
                ]
                pred_df_filtered = pred_df.loc[filt].copy()

                if pred_df_filtered['raw_confs'].sum() == 0:
                    pred_df_filtered['confs'] = 1 / len(pred_df_filtered)
                else:
                    pred_df_filtered['confs'] = (
                        pred_df_filtered['raw_confs'] /
                        pred_df_filtered['raw_confs'].sum()
                    )

            pred_df_show = pred_df_filtered[
                pred_df_filtered['confs'] >= threshold
            ].head(n_classes)
            pred_df_show['confs'] = (
                100*pred_df_show['confs']
            ).round(0).astype(int)

            prediction_l[idx] = {
                'description': desc_l[idx],
                'original_code': original_code_l[idx],
                'label': tuple(pred_df_show['labels'].tolist()),
                'confidence': tuple(pred_df_show['confs'].tolist()),
                'hierarchical_level': hierarchical_level,
                'identifier': identifier_l[idx],
                'title': tuple([
                    self.structure.titles[label]
                    for label in pred_df_show['labels'].tolist()
                ])
            }

        return prediction_l
