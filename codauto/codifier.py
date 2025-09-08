# -*- coding: utf-8 -*-
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

from codauto.utils import truncate_colormap
from codauto.utils import preprocess_text
from codauto.utils import forward_pseudolog_transform
from codauto.utils import reverse_pseudolog_transform


class Codifier(ABC):
    """
    Abstract base class for hierarchical codifiers.

    This class handles the preprocessing, cleaning, and validation of
    training and testing datasets for hierarchical classification tasks.
    It also manages correspondences between previous and new classification
    codes and provides methods for predictions and evaluation.

    Attributes:
        root_path (str): Path to store temporary files and models.
        logger (logging.Logger): Logger for the instance.
        min_lenght_texts (int): Minimum length of valid descriptions.
        structure: Object containing the hierarchical structure of codes.
        model: Placeholder for the trained model.
        correspondences: Dictionary mapping old codes to new codes.
        train_df (pd.DataFrame): Cleaned training dataset.
        test_df (pd.DataFrame): Cleaned testing dataset with hierarchical labels.

    Methods:
        get_correspondences(corres_df):
            Converts a correspondence DataFrame into a hierarchical dictionary.

        get_code(n):
            Standardizes numeric codes to 4-character strings and checks validity.

        clean_data(data_df):
            Cleans and preprocesses the dataset codes and descriptions.

        load_data(data_df, data_name='data_df'):
            Loads and cleans a dataset, reporting bad or missing codes/descriptions.

        check_codes(data_df, name, show_codes=True):
            Checks which codes from the hierarchy are missing in the dataset.

        get_train_dataset(train_dataset, name):
            Cleans and validates the training dataset.

        get_test_dataset(test_dataset, name):
            Cleans and validates the testing dataset and generates hierarchical ground-truth labels.

        train(**kwargs):
            Abstract method to train the model on the training dataset.

        save(name):
            Abstract method to save the trained model.

        load(name):
            Abstract method to load a previously trained model.

        get_pred_for_batch(samples, idxs, clean_samples=False):
            Abstract method to generate predictions for a batch of samples.

        get_top_n_predictions(samples, idxs, n_classes=5, clean_samples=False):
            Returns the top N predicted labels with confidences for a batch.

        get_preds_test_set(test_set, n_classes):
            Generates top-N predictions for the entire test dataset.

        get_precision_vs_recall_multi_conf(test_set, version, n_classes, source='all'):
            Computes precision, recall, and average predicted classes across confidence thresholds.

        plot_evaluate_curve(test_set, source='all', version='09'):
            Plots precision-recall curves for the test set for different thresholds.

        evaluate(source='all', get_curve=True, simplify=True, version='model'):
            Evaluates the model on the test set and returns metrics and/or classification report.

        predict(desc_l, mode, hierarchical_level, threshold, original_code_l=None, identifier_l=None):
            Predicts codes for a list of descriptions, optionally considering direct recoding correspondences.
    """

    def __init__(self,
                 structure_instance,
                 train_df,
                 test_df,
                 root_path,
                 corres_df=None,
                 min_lenght_texts=3
                 ):
        self.root_path = root_path
        os.makedirs(self.root_path, exist_ok=True)
        self.logger = logging.getLogger(f'CNAECodifier.{id(self)}')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(stream_handler)
        self.logger.propagate = False
        self.min_lenght_texts = min_lenght_texts
        self.structure = structure_instance
        self.model = None
        if corres_df is not None:
            self.correspondences = self.get_correspondences(corres_df)
        else:
            self.correspondences = None
        self.train_df = self.get_train_dataset(train_df, 'train_df')
        self.test_df = self.get_test_dataset(test_df, 'test_df')

    def get_correspondences(self, correspondences_df):
        """
        Processes a DataFrame containing code correspondences and organizes them
        by hierarchical level.

        Parameters
        ----------
        correspondences_df : pandas.DataFrame
            A DataFrame with exactly 3 columns:
            [0] previous classification code (old_code)
            [1] new classification code (new_code)
            [2] hierarchical level of the correspondence (level)

        Returns
        -------
        pandas.Series
            A Series grouped by hierarchical level. Each element is a mapping
            from old codes to a set of corresponding new codes at that level.

        Raises
        ------
        ValueError
            If the input DataFrame does not have exactly 3 columns.

        Notes
        -----
        - The method cleans the codes by removing periods.
        - The hierarchical levels are converted to integers.
        - The output allows easy lookup of how old codes map to new codes
          for each level of the hierarchy.
        """
        if correspondences_df.shape[1] != 3:
            raise ValueError(
                "corres_df must have 3 columns: [0] previous classification code, [1] new classification code, [2]  hierarchical level of the correspondence")
        correspondences_df.columns = ['old_code', 'new_code', 'level']
        correspondences_df = correspondences_df.map(
            lambda v: v.replace('.', '')
        )
        correspondences_df['level'] = correspondences_df['level'].apply(int)
        correspondences = correspondences_df.groupby('level').apply(
            lambda df: df.groupby('old_code').new_code.apply(set)
        )

        return correspondences

    def get_code(self, code):
        """
        Returns the code if it exists in the reversed hierarchy; otherwise returns NaN.

        This method checks whether a given code is present in the object's
        reversed hierarchy (`reversed_hierarchy`). If the code exists, it is
        returned as-is; if not, `np.nan` is returned.

        Parameters
        ----------
        code : str
            The code to check in the reversed hierarchy.

        Returns
        -------
        str or float
            - `str`: the same code if it exists in `reversed_hierarchy`.
            - `np.nan`: if the code is not found.

        Notes
        -----
        - `self.structure.reversed_hierarchy` is expected to be a dictionary containing all valid codes.
        - This method is useful for validating codes before processing or adding them to normalized predictions.
        """
        if code in self.structure.reversed_hierarchy.keys():
            return code
        return np.nan

    def clean_data(self, data_df):
        """
        Cleans and preprocesses a dataset by standardizing codes and text descriptions.

        Parameters
        ----------
        data_df : pandas.DataFrame
            The input dataframe with at least two columns:
            - Column 0: code
            - Column 1: text description

        Returns
        -------
        pandas.DataFrame
            The cleaned dataframe with:
            - Standardized 4-character codes in the first column.
            - Preprocessed text descriptions in the second column.
            - Texts shorter than the minimum length replaced with NaN.

        Processing Steps
        ----------------
        1. Extracts the first two columns of the dataframe.
        2. Standardizes the codes in the first column using droping dots and get_code.
        3. Applies `preprocess_text` to clean the text in the second column.
        4. Replaces text entries that are shorter than `min_lenght_texts`
           with NaN.
        """
        col_l = data_df.columns
        col_n_0 = col_l[0]
        col_n_1 = col_l[1]
        data_df[col_n_0] = data_df[col_n_0].apply(
            lambda n: self.get_code(str(n).replace('.', ''))
        )
        data_df[col_n_1] = data_df[col_n_1].apply(preprocess_text)
        data_df[col_n_1] = data_df[col_n_1].apply(
            lambda desc: np.nan if len(desc) <= self.min_lenght_texts else desc
        )
        return data_df

    def load_data(self, data_df, data_name='data_df'):
        """
        Loads and preprocesses a dataset, cleaning it and removing invalid entries.

        Parameters
        ----------
        data_df : pandas.DataFrame
            Input dataframe to be cleaned and loaded.
        data_name : str, optional
            Name of the dataset (used for logging), by default 'data_df'.

        Returns
        -------
        pandas.DataFrame
            The cleaned dataframe with all invalid codes or descriptions removed,
            indices reset, and ready for further processing.

        Process Overview
        ----------------
        1. Logs the start of the loading process.
        2. Copies the input dataframe to avoid modifying the original.
        3. Logs the raw number of entries in the dataset.
        4. Cleans the dataframe using the `clean_data` method:
           - Standardizes codes.
           - Preprocesses text descriptions.
           - Replaces too-short descriptions with NaN.
        5. Counts and logs the number of invalid codes and descriptions (NaN values).
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
        Checks which hierarchical codes are missing in a given dataset.

        Parameters
        ----------
        data_df : pandas.DataFrame
            The dataset to check. Assumes that the first column contains the codes.
        name : str
            Name of the dataset (used for logging purposes).
        show_codes : bool, optional
            Whether to log the list of unrepresented codes, by default True.

        Returns
        -------
        int
            The number of codes in the hierarchy that are not represented in the dataset.

        Process Overview
        ----------------
        1. Extracts the unique codes present in the dataset.
        2. Retrieves all codes defined in the hierarchical structure.
        3. Computes the set difference to find unrepresented codes.
        4. Logs the number of missing codes and optionally the list of these codes.
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
        Prepares and validates the training dataset for the codifier model.

        Parameters
        ----------
        train_dataset : pandas.DataFrame
            The raw training dataset. Expected to have at least two columns:
            [0] code, [1] description. Extra columns will be ignored.
        name : str
            Name of the dataset, used for logging and warnings.

        Returns
        -------
        pandas.DataFrame
            A cleaned and validated training dataset containing only valid codes
            and descriptions.

        Process Overview
        ----------------
        1. Checks that the dataset has at least 2 columns. Raises an error otherwise.
        2. Warns if the dataset has more than 2 columns, but only uses the first two.
        3. Cleans the dataset using `load_data` (removes invalid codes/descriptions).
        4. Checks for unrepresented codes using `check_codes`.
        5. Issues a warning if some codes in the hierarchy are missing from the dataset.
        6. Returns the cleaned dataset ready for training.
        """

        if train_dataset.shape[1] < 2:
            raise ValueError("train_df must have at least 2 columns: [0] code, [1] description")
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
        Prepares and validates the test dataset for hierarchical evaluation.

        Parameters
        ----------
        test_dataset : pandas.DataFrame
            Raw test dataset. Must have exactly three columns:
            [0] code, [1] description, [2] source.
        name : str
            Name of the dataset, used for logging and warnings.

        Returns
        -------
        pandas.DataFrame
            Processed test dataset containing:
            - 'desc': cleaned description text.
            - 'source': source of the sample.
            - 'gt_<level>': ground-truth code at each hierarchical level.

        Method Details
        --------------
        1. Validates that the input DataFrame has exactly 3 columns; raises ValueError if not.
        2. Cleans the dataset using `load_data` to handle invalid or too short codes/descriptions.
        3. Renames columns to standard format:
           - Top-level code: 'gt_<highest_level>'
           - Description: 'desc'
           - Source: 'source'
        4. Expands the top-level code into ground-truth codes for all hierarchical levels
           using `self.structure.reversed_hierarchy`.
        5. Returns a DataFrame ready for hierarchical evaluation.
        """
        if test_dataset.shape[1] != 3:
            raise ValueError("test_df must have 3 columns: [0] code, [1] description, [2] source")
        input_df = self.load_data(test_dataset, name)
        input_df.columns = [f'gt_{self.structure.level_l[-1]}', 'desc', 'source']
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
        This method must be implemented in any subclass of Codifier. It should handle
        the actual training procedure of the underlying model, storing the trained model
        in `self.model`.
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
        This method must be implemented in any subclass of Codifier. It should handle
        serialization of the trained model (`self.model`) so that it can be reloaded
        later using the `load` method.
        """

    @abstractmethod
    def load(self, name):
        """
        Abstract method to load a previously saved model from disk.

        Parameters
        ----------
        name : str
            The filename (or path) from which the trained model should be loaded.

        Notes
        -----
        This method must be implemented in any subclass of Codifier. It should handle
        deserialization of the model and assign it to `self.model` so that the codifier
        can be used for predictions.
        """

    @abstractmethod
    def get_pred_for_batch(self, samples, idxs, clean_samples=False):
        """
        Abstract method to generate predictions for a batch of input samples.

        Parameters
        ----------
        samples : list of str
            A list of text descriptions to be predicted/classified.
        idxs : list of int
            A list of indices corresponding to each sample in the batch.
        clean_samples : bool, optional (default=False)
            Whether to apply preprocessing/cleaning to the samples before prediction.

        Returns
        -------
        dict
            A dictionary where keys are indices from `idxs` and values are dictionaries
            containing predicted labels and their associated confidence scores for each
            hierarchical level.

        Notes
        -----
        Subclasses must implement this method to perform the actual batch prediction.
        The output format should match what `get_top_n_predictions` expects for further processing.
        """

    def get_top_n_predictions(self, samples, idxs, n_classes=5, clean_samples=False):
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
            Whether to apply preprocessing/cleaning to the samples before prediction.

        Returns
        -------
        dict
            A dictionary where each key is an index from `idxs`, and each value is another
            dictionary containing the top-N predicted labels and their corresponding confidence
            scores for each hierarchical level. Keys follow the format:
            - 'label_<level>': list of top-N predicted labels
            - 'conf_<level>': list of corresponding confidence scores

        Notes
        -----
        This method calls `get_pred_for_batch` to obtain raw predictions, then filters
        them to only keep the top-N predictions per level. It is useful for scenarios
        where only the most likely predictions are needed, such as displaying suggestions
        or computing metrics.
        """
        hierarchical_preds = self.get_pred_for_batch(samples, idxs, clean_samples)
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
         Generate predictions for an entire test dataset using the top-N prediction strategy.

         Parameters
         ----------
         test_set : pandas.DataFrame
             A DataFrame containing the test samples. Must include a column 'desc' with text descriptions.
         n_classes : int
             The number of top predictions to return for each sample and hierarchical level.

         Returns
         -------
         pandas.DataFrame
             A DataFrame where each row corresponds to a sample from `test_set`, indexed by
             the original DataFrame index. Each column contains the top-N predicted labels and
             confidence scores per hierarchical level.

         Notes
         -----
         - Iterates over each row of the test dataset.
         - Calls `get_top_n_predictions` for each description individually.
         - Collects predictions in a dictionary and converts it to a DataFrame for easier analysis
           and merging with ground truth labels.
         - Useful for evaluating the model across the full test dataset or generating prediction reports.
         """
        preds_test = dict()
        for row_i, row in tqdm(
                test_set.iterrows(), total=len(test_set), desc=f"Predicting test_set for {n_classes} classes"
        ):
            sample_pred = self.get_top_n_predictions(
                [row['desc']], [row_i], n_classes
            )
            preds_test[row_i] = sample_pred[row_i]
        return pd.DataFrame.from_dict(preds_test, orient='index')

    def get_precision_vs_recall_multi_conf(
            self, test_set, version, n_classes, source='all'
            ):
        """
        Compute precision, recall, and average number of predicted classes across multiple confidence thresholds
        for each hierarchical level.

        Parameters
        ----------
        test_set : pandas.DataFrame
            The test dataset containing ground truth labels and descriptions.
        version : str
            A string identifier for the model version or evaluation run (used for logging or plotting purposes).
        n_classes : int
            The number of top predictions to consider for evaluation.
        source : str, default 'all'
            If not 'all', filters the test set to include only rows from this source.

        Returns
        -------
        dict
            A dictionary where keys are hierarchical levels and values are DataFrames indexed by confidence
            thresholds. Each DataFrame contains the following columns:
                - 'precision': Fraction of predictions above the threshold that are correct.
                - 'recall': Fraction of samples with at least one prediction above the threshold.
                - 'mean_classes': Average number of predicted classes above the threshold.

        Notes
        -----
        - Drops test samples with missing ground truth for the deepest hierarchical level.
        - Iterates through confidence thresholds from 0 to 1 with step 0.0025.
        - Computes whether the ground truth label is present among predictions exceeding the threshold.
        - Useful for generating precision-recall curves and analyzing model confidence behavior
          across hierarchical levels.
        """
        column = f'gt_{self.structure.level_l[-1]}'
        test_set.dropna(subset=[column], inplace=True)

        if source != 'all':
            if source not in test_set['source'].values:
                raise ValueError(
                    f'The source value {source} is not present in the source column.')
            test_set = test_set[test_set['source'] == source]

        preds_test_set = self.get_preds_test_set(test_set, n_classes)
        df_gt = test_set.merge(
            preds_test_set,
            left_index=True, right_index=True,
        )
        prec_recall_mean_classes_d = dict()
        for lvl in self.structure.level_l:
            prec_recall_mean_classes = dict()
            for th in np.arange(0, 1.001, 0.0025):
                n_classes_th = []
                for conf_list in df_gt[f'conf_{lvl}']:
                    conf_list = np.array(conf_list)
                    n_classes_th.append((conf_list > th).sum())
                n_classes_mean = np.mean([x for x in n_classes_th if x > 0.1])
                df_gt[f'gt_vs_pl_{lvl}'] = df_gt.apply(
                    lambda row: (
                        str(row[f'gt_{lvl}']) in row[f'label_{lvl}'] and
                        float(row[f'conf_{lvl}'][row[f'label_{lvl}'].index(
                            str(row[f'gt_{lvl}']))]) > th
                    ) if str(row[f'gt_{lvl}']) in row[f'label_{lvl}'] else False,
                    axis=1
                )
                df_gt_th = df_gt[df_gt[f'conf_{lvl}'].apply(
                    lambda lista_conf: any(np.array(lista_conf) > th))]
                n_correct_samples = df_gt_th[f'gt_vs_pl_{lvl}'].sum()
                n_samples = len(df_gt_th)
                recall = n_samples / len(df_gt)
                if n_samples > 0:
                    precision = n_correct_samples / n_samples
                else:
                    precision = 0
                prec_recall_mean_classes[th] = {
                    'precision': precision, 'recall': recall, 'mean_classes': n_classes_mean}
            prec_recall_mean_classes = pd.DataFrame.from_dict(
                prec_recall_mean_classes, orient='index').dropna()
            prec_recall_mean_classes_d[lvl] = prec_recall_mean_classes

        return prec_recall_mean_classes_d

    def plot_evaluate_curve(self, test_set, source='all', version='09'):
        """
        Plot precision vs recall curves for different numbers of predicted classes (1 and 15)
        across all hierarchical levels, highlighting specific confidence thresholds.

        Parameters
        ----------
        test_set : pandas.DataFrame
            The test dataset containing ground truth labels, descriptions, and source.
        source : str, default 'all'
            If not 'all', filters the test set to include only rows from this source.
        version : str, default '09'
            A string identifier for the model version or evaluation run, used in plot titles.

        Notes
        -----
        - Computes precision and recall using `get_precision_vs_recall_multi_conf` for n_classes=1 and n_classes=15.
        - Highlights four confidence thresholds: 0.8, 0.25, 0.1, and 0.03 on the curves.
        - Uses color and marker size to visualize the average number of predicted output codes.
        - Applies a pseudologarithmic x-scale to improve visualization for high-recall regions.
        - Displays a separate plot for each hierarchical level in the structure.
        - Provides logging when levels or thresholds are missing in the computed precision-recall data.

        Visualization
        -------------
        - Scatter plot for n_classes=15 uses a color map representing the average number of output codes.
        - Scatter plot for n_classes=1 uses black markers.
        - Blue '+' markers indicate selected confidence thresholds on the curves.
        """
        precision_vs_recall_multi_conf_15 = self.get_precision_vs_recall_multi_conf(
            test_set, version=version, n_classes=15, source=source
        )
        precision_vs_recall_multi_conf_1 = self.get_precision_vs_recall_multi_conf(
                test_set, version=version, n_classes=1, source=source
        )
        THRESHOLDS = (0.8, 0.25, 0.1, 0.03)
        cmap = plt.get_cmap('gist_stern')
        new_cmap = truncate_colormap(cmap, 0, 0.8)

        for lvl in self.structure.level_l:
            if lvl not in precision_vs_recall_multi_conf_15 or lvl not in precision_vs_recall_multi_conf_1:
                self.logger.info(f"Level {lvl} not found in precision_vs_recall data.")
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
                            precision_vs_recall[lvl].loc[th, 'recall'] + 0.0005,
                            precision_vs_recall[lvl].loc[th, 'precision'] + 0.003,
                            f'{th}',
                            size=9,
                            color='blue'
                        )
                    else:
                        self.logger.info(
                            f"Threshold {th} not found for level {lvl} in precision_vs_recall_multi_conf_{n}.")
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
                f'Precision vs Recall for {version} {lvl}', fontsize=18)
            ax.grid(True)
            plt.show()

    def evaluate(self,
                 source='all',
                 get_curve=True,
                 simplify=True,
                 version='model'
                 ):
        """
        Evaluate the model's performance on the test dataset by computing metrics and optionally plotting curves.

        Parameters
        ----------
        source : str, default 'all'
            If not 'all', filters the test set to include only rows from this source.
        get_curve : bool, default True
            If True, plots precision vs recall curves using `plot_evaluate_curve`.
        simplify : bool, default True
            If True, returns only a dictionary of metrics (accuracy, F1 scores).
            If False, also returns a detailed classification report for each hierarchical level.
        version : str, default 'model'
            Identifier for the evaluation run, used in plot titles.

        Returns
        -------
        dict
            If `simplify=True`, returns a dictionary containing accuracy and F1 scores (micro, macro, weighted) per hierarchical level.
        tuple
            If `simplify=False`, returns a tuple `(metrics, report)` where `report` contains the detailed classification reports per level.

        Notes
        -----
        - Filters the test set based on the provided source.
        - Checks for missing or unrepresented codes in the test set using `check_codes`.
        - Generates predictions for the test set with `get_preds_test_set`.
        - Computes standard evaluation metrics for each hierarchical level.
        - If `get_curve` is True, calls `plot_evaluate_curve` to visualize precision vs recall for n_classes=1 and 15.
        - Uses zero_division=0 for F1 scores to handle cases where a class has no predicted samples.
        """
        if source != 'all':
            if source not in self.test_df['source'].values:
                raise ValueError(
                    f'The source value {source} is not present in the source column.')
            test_set = self.test_df[self.test_df['source'] == source]
        else:
            test_set = self.test_df.copy()
        self.logger.info(
            f'Test set size {len(test_set)} for source {source}')
        self.check_codes(test_set, f'test_set_{source}', False)
        # Se ejecuta get_preds_test_set
        preds_test_set = self.get_preds_test_set(
            test_set=test_set, n_classes=1)
        df_gt = test_set.merge(
            preds_test_set, left_index=True, right_index=True)
        metrics = dict()
        report = dict()
        if get_curve:
            self.plot_evaluate_curve(
                test_set=test_set,
                version=version
            )
        for lvl in self.structure.level_l:
            df_gt_list = df_gt[f'gt_{lvl}'].astype(str).tolist()
            pred_label = [label[0] for label in df_gt[f'label_{lvl}']]
            accuracy = accuracy_score(df_gt_list, pred_label)
            f1_micro = f1_score(df_gt_list, pred_label,
                                average='micro', zero_division=0)
            f1_macro = f1_score(df_gt_list, pred_label,
                                average='macro', zero_division=0)
            f1_weighted = f1_score(df_gt_list, pred_label,
                                   average='weighted', zero_division=0)
            metrics[lvl] = {
                'accuracy': accuracy,
                'f1_score': {
                    'micro': f1_micro,
                    'macro': f1_macro,
                    'weighted': f1_weighted
                }
            }
            if not simplify:
                report[lvl] = classification_report(
                    df_gt_list, pred_label, labels=sorted(list(set(df_gt_list))))

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
            Original codes for direct recoding. If provided, predictions may be overridden
            by direct correspondences for codes that map unambiguously.
        identifier_l : list, optional
            Unique identifiers corresponding to each description. Defaults to sequential integers.

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
        - Handles direct recoding using `self.correspondences` if `original_code_l` is provided.
        - For non-direct recoding, predictions are obtained from `get_pred_for_batch`.
        - Confidence scores are normalized if original codes restrict possible labels.
        - Predictions are filtered based on the confidence threshold and limited to `n_classes` according to mode.
        """
        if identifier_l is None:
            identifier_l = list(range(len(desc_l)))
        if mode == "codification":
            n_classes = 1
        elif mode == "assistance":
            n_classes = 15
        else:
            raise ValueError

        hierarchical_level_name = self.structure.level_l[hierarchical_level - 1]

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
                            'title': (self.structure.titles[tuple(filt)[0]])
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

        raw_predictions = self.get_pred_for_batch(
            desc_l_not_direct_recoding,
            idxs_not_direct_recoding,
            True
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
