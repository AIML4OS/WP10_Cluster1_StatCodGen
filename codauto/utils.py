# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:21:09 2024

@author: git.metodologia@ine.es
"""

import re
import unicodedata
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors


def preprocess_text(input_str):
    """
    Cleans and normalizes the input text by:
    - Converting to lowercase
    - Removing unnecessary spaces and accents, while preserving 'ñ'.
    - Keeping only letters a-z, spaces, and 'ñ'

    The result is returned as a UTF-8 encoded string.

    Args:
        input_str (str): Input text to preprocess.
    Returns:
        str: Cleaned, normalized UTF-8 string.
    """
    output_str = str(input_str).lower()
    output_str = output_str.replace('ñ', '__enie__')
    output_str = unicodedata.normalize('NFD', output_str)
    output_str = output_str.encode('ascii', 'ignore').decode('utf-8')
    output_str = output_str.replace('__enie__', 'ñ')
    output_str = re.sub(r'[^a-z ñ]', '', output_str)
    return ' '.join(output_str.split())


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Creates a truncated version of a given matplotlib colormap.

    This function is useful for visualizations where only a portion of the colormap 
    is desired (e.g., to emphasize lower or higher values in a heatmap).

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The original colormap to be truncated.

    minval : float, optional (default=0.0)
        The lower bound (between 0 and 1) of the original colormap to include.

    maxval : float, optional (default=1.0)
        The upper bound (between 0 and 1) of the original colormap to include.

    n : int, optional (default=100)
        The number of discrete color levels to generate in the truncated colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A new colormap object that includes only the specified segment of the original colormap.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(
            n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def forward_pseudolog_transform (x):
    """
    Applies a forward pseudo-logarithmic transformation to the input.

    The transformation is defined as: `-log(-x + 1.01)`. It is typically used
    to compress values into a logarithmic scale, while avoiding singularities
    near `x = 1.0` due to the `+ 1.01` offset.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s) to transform. Values should be strictly less than 1.01 to avoid invalid output.

    Returns
    -------
    float or numpy.ndarray
        Transformed value(s) in the compressed log scale.
    """
    return -np.log(-x + 1.01)


def reverse_pseudolog_transform(x):
    """
    Applies the inverse transformation of `forward`.

    The transformation is defined as: `1 - exp(-x)`, which reverses the effect 
    of the `forward` method and maps log-scale values back to the original range.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s) previously transformed using `forward`.

    Returns
    -------
    float or numpy.ndarray
        Reconstructed value(s) in the original scale.
    """
    return 1.01 - np.exp(-x)


def get_combined_evaluate_curve(
                    pre_recall_mean0_15,
                    pre_recall_mean0_1,
                    pre_recall_mean1_15,
                    pre_recall_mean1_1,
                    levels=['section', 'division', 'group', 'class'],
                    name_0='CNAE09',
                    name_1='CNAE25'
  ):
    """
    Plot combined Precision vs Recall curves for two hierarchical codification systems across multiple levels.

    This function compares and visualizes the performance of two codification systems (e.g., CNAE09 vs CNAE25)
    using Precision vs Recall scatter plots. The visualization includes different configurations based on 
    the number of predictions (top-1 vs top-15), average number of predicted classes, and confidence thresholds.

    Parameters
    ----------
    pre_recall_mean0_15 : dict
        Precision-recall data for system `name_0` using top-15 predictions. Must be structured by levels.
    pre_recall_mean0_1 : dict
        Precision-recall data for system `name_0` using top-1 predictions.
    pre_recall_mean1_15 : dict
        Precision-recall data for system `name_1` using top-15 predictions. Must be structured by levels.
    pre_recall_mean1_1 : dict
        Precision-recall data for system `name_1` using top-1 predictions.
    levels : list of str, optional
        Hierarchical levels to plot (e.g., ['section', 'division', 'group', 'class']).
    name_0 : str, optional
        Name of the first codification system, used for labeling (default is 'CNAE09').
    name_1 : str, optional
        Name of the second codification system, used for labeling (default is 'CNAE25').
    Notes
    -----
    - Each scatter point represents a precision/recall value with color and size indicating the average 
      number of predicted classes.
    - Custom x-axis scale is applied using log-like transformation via `forward` and `reverse` functions 
      from the codifier.
    - Confidence thresholds (e.g., 0.8, 0.25, 0.1, 0.03) are annotated with blue "+" markers.

    Returns
    -------
    None
        The function displays or saves the plots but does not return any value.

    Raises
    ------
    ValueError
        If required level data is missing or the codifier lacks required methods.

    Examples
    --------
    >>> get_combined_evaluate_curve(
            pr_cnae09_15, pr_cnae09_1,
            pr_cnae25_15, pr_cnae25_1,
            levels=['group', 'class'],
            name_0='CNAE09', name_1='CNAE25',
            save_path='./plots'
        )
    """
    THRESHOLDS = (0.8, 0.25, 0.1, 0.03)
    cmap = plt.get_cmap('gist_stern')
    new_cmap = truncate_colormap(cmap, 0, 0.8)

    for lvl in levels:
        if lvl not in pre_recall_mean0_15 or lvl not in pre_recall_mean0_1:
            print(f"Level {lvl} not found in precision_vs_recall_{name_0} data.")
            continue
        if lvl not in pre_recall_mean1_15 or lvl not in pre_recall_mean1_1:
            print(f"Level {lvl} not found in precision_vs_recall_{name_1} data.")
            continue

        fig, ax = plt.subplots(figsize=(15, 6))

        scatter_0 = ax.scatter(
            pre_recall_mean0_15[lvl]['recall'],
            pre_recall_mean0_15[lvl]['precision'],
            c=pre_recall_mean0_15[lvl]['mean_classes'],
            cmap=new_cmap,
            s=3 * pre_recall_mean0_15[lvl]['mean_classes'] ** 2,
            marker='o',
            alpha=0.2
        )
        fig.colorbar(scatter_0, label=f'Avg. Num. Output Codes - {name_0}')

        scatter_1 = ax.scatter(
            pre_recall_mean1_15[lvl]['recall'],
            pre_recall_mean1_15[lvl]['precision'],
            c=pre_recall_mean1_15[lvl]['mean_classes'],
            cmap=new_cmap,
            s=3 * pre_recall_mean1_15[lvl]['mean_classes'] ** 2,
            marker='o'
        )
        fig.colorbar(scatter_1, label=f'Avg. Num. Output Codes - {name_1}')

        for th in THRESHOLDS:
            if th in pre_recall_mean1_15[lvl].index:
                ax.scatter(
                    pre_recall_mean1_15[lvl].loc[th, 'recall'],
                    pre_recall_mean1_15[lvl].loc[th, 'precision'],
                    color='blue', marker='+', s=80
                )
                ax.text(
                    pre_recall_mean1_15[lvl].loc[th, 'recall'] + 0.0005,
                    pre_recall_mean1_15[lvl].loc[th, 'precision'] + 0.003,
                    f'{th}', size=9, color='blue'
                )
            else:
                print(
                    f"Threshold {th} not found for level {lvl} in precision_vs_recall_multi_conf_15.")

        scatter_0_1 = ax.scatter(
            pre_recall_mean0_1[lvl]['recall'],
            pre_recall_mean0_1[lvl]['precision'],
            c='black',
            s=3 * pre_recall_mean0_1[lvl]['mean_classes'] ** 2,
            marker='o',
            alpha=0.08
        )

        scatter_1_1 = ax.scatter(
            pre_recall_mean1_1[lvl]['recall'],
            pre_recall_mean1_1[lvl]['precision'],
            c='black',
            s=3 * pre_recall_mean1_1[lvl]['mean_classes'] ** 2,
            marker='o'
        )

        for th in THRESHOLDS:
            if th in pre_recall_mean1_1[lvl].index:
                ax.scatter(
                    pre_recall_mean1_1[lvl].loc[th, 'recall'],
                    pre_recall_mean1_1[lvl].loc[th, 'precision'],
                    color='blue', marker='+', s=80
                )
                ax.text(
                    pre_recall_mean1_1[lvl].loc[th, 'recall'] + 0.0005,
                    pre_recall_mean1_1[lvl].loc[th, 'precision'] + 0.003,
                    f'{th}', size=9, color='blue'
                )
            else:
                print(
                    f"Threshold {th} not found for level {lvl} in precision_vs_recall_multi_conf_1.")

        ax.set_xlim((0, 1.002))
        ax.set_ylim((0.4, 1.01))
        ax.set_xscale('function', functions=(
            forward_pseudolog_transform, reverse_pseudolog_transform
        ))
        ax.set_xticks([0.2, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 0.997, 1])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(
            f'Precision vs Recall for {name_0}-{name_1} {lvl}', fontsize=18)
        ax.grid(True)
        plt.show()
