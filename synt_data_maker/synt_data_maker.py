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

from tqdm import tqdm
from datetime import datetime
import os
import pandas as pd
from abc import ABC, abstractclassmethod
import requests
from collections import Counter, defaultdict
import statistics as stats
import re
from groq import Groq, RateLimitError
from openai import AzureOpenAI

from data_handler import read_json
from prompt_synt_data_maker import PromptSyntDataMaker


class SyntDataGenerator(ABC):
    """
    Initialize a synthetic data generator backed by an LLM and a prompt builder.

    This constructor stores configuration, loads (static) examples from disk,
    instantiates a prompt builder, and optionally enriches each entry in
    `label_contents_map_dict` with examples depending on the selected example strategy.

    Parameters
    ----------
    root : str or os.PathLike
        Project root directory. Used to locate inputs (e.g., example files).
    api_key : str or None
        API key used by the concrete backend implementation.
    model : str
        Model identifier or deployment name used by the backend.
    label_contents_map_dict: dict
        Mapping of class/label identifiers to dictionaries that must contain
        at least a ``'description'`` field. This mapping is mutated in-place
        later (e.g., adding ``'prompt'`` and ``'response'``).
    body_intro : str
        Introductory text included in prompts.
    task : str, optional
        Task identifier used to select examples from `eg_filename`.
    n_responses : int, optional
        Number of responses requested to the model (used inside prompts).
    p_lang : str, default='en'
        Prompt language passed to the prompt builder.
    o_lang : str, optional
        Output language requested in the prompt.
    output_format: str, optional
        Output format requested in the prompt (note: parameter name is `output_format`
        in the current code).
    len_constraint : str or int, optional
        Length/longitude constraint used inside prompts.
    eg_filename : str, default='static_examples.json'
        Filename (under ``{root}/input``) with static examples in JSON format.
    eg_strategy : {'static', 'mixed', None}, default='static'
        Example strategy:
        - ``'static'``: always inject static examples for the selected task.
        - ``'mixed'``: only inject static examples when an entry has no example.

    Attributes
    ----------
    date : str
        Current date formatted as ``'%d%m%Y'``.
    synth_data_df : pandas.DataFrame or None
        Cached synthetic dataset produced by :meth:`get_synt_data_df`.
    prompt_fields : dict
        Dictionary of prompt fields passed to the prompt builder.

    Notes
    -----
    This class is abstract. Subclasses must implement :meth:`api_call`.
    """
    def __init__(
        self,
        root,
        api_key,
        model,
        label_contents_map_dict,
        body_intro=None,
        task=None,
        n_responses=None,
        p_lang='en',
        o_lang=None,
        output_format=None,
        len_constraint=None,
        eg_filename='static_examples.json',
        eg_strategy='static'
    ):
        self.date = datetime.now().strftime('%d%m%Y')
        self.root = root
        # data
        self.label_contents_map_dict= label_contents_map_dict
        self.synth_data_df = None
        # prompt data
        self.prompt_fields = {
            'body_intro': body_intro,
            'task': task,
            'number': n_responses,
            'language': o_lang,
            'format': output_format,
            'len_constraint': len_constraint
        }
        self.prompt_maker = PromptSyntDataMaker(root, prompt_lang=p_lang)
        self.eg_filename = read_json(
            os.path.join(self.root, 'input', eg_filename)
        )
        self.eg_strategy = eg_strategy
        self.model = model
        self.api_key = api_key
        self.get_examples()

    def get_examples(self):
        """
        Populate example prompts in `label_contents_map_dict` according to the configured strategy.

        If a `task` is set, this method retrieves examples from the loaded JSON
        (`eg_filename`) and injects them into each entry of `label_contents_map_dict` under the key
        ``'example'``:

        - If ``eg_strategy == 'static'``: always inject the example for the selected task.
        - If ``eg_strategy == 'mixed'``: inject only when an entry has no ``'example'``.

        Returns
        -------
        None
            The method updates `self.label_contents_map_dict` in-place.
        """
        task = self.prompt_fields.get('task')
        if task and self.eg_strategy == 'static':
            for data_d in self.label_contents_map_dict.values():
                data_d['example'] = self.eg_filename[task]

        if task and self.eg_strategy == 'mixed':
            for data_d in self.label_contents_map_dict.values():
                if not data_d.get('example'):
                    data_d['example'] = self.eg_filename[task]

    def api_call(self, prompt):
        """
        Call the underlying LLM backend to generate text from a prompt.

        Parameters
        ----------
        prompt : str
            Fully rendered prompt to send to the model.

        Returns
        -------
        str
            Model-generated content as plain text.

        Notes
        -----
        Subclasses must implement this method.
        """
        pass

    def set_api_responses(self):
        self.prompt_maker.logger.info('Setting api responses...')
        for data_d in tqdm(self.label_contents_map_dict.values(), total=len(self.label_contents_map_dict.values())):
            self.prompt_fields['description'] = data_d.get('description')
            if self.eg_strategy:
                self.prompt_fields['example'] = data_d.get('example')
            prompt = self.prompt_maker.get_prompt(self.prompt_fields)
            data_d['prompt'] = prompt
            data_d['response'] = self.api_call(prompt)

    def get_synt_data_df(self):
        """
        Generate prompts for each entry and store the LLM responses in `label_contents_map_dict`.

        For each value in `self.label_contents_map_dict`, this method:
        1) injects the entry description (and optionally an example) into `self.prompt_fields`,
        2) builds a prompt using the prompt builder,
        3) stores the prompt under ``data_d['prompt']``,
        4) calls :meth:`api_call` and stores the result under ``data_d['response']``.

        Returns
        -------
        synth_data_df: pandas.DataFrame
        Pandas DataFrame containing labeled model generate data.

        Notes
        -----
        This method performs network calls via the concrete backend implementation.
        """
        index_ = 0
        row_dict = {}
        self.set_api_responses()
        self.prompt_maker.logger.info('Getting response df...')
        for label, data_d in tqdm(self.label_contents_map_dict.items(), total=len(self.label_contents_map_dict.keys())):
            for sentence in data_d['response'].split('\n'):
                row_dict[index_] = {
                    'label': label,
                    'literal': sentence
                }
                index_ += 1
        self.synth_data_df = pd.DataFrame.from_dict(row_dict, orient='index')
        return self.synth_data_df

    @staticmethod
    def clean_data_df(data_df, label_col, text_col):
        """
        Build and return a DataFrame of synthetic examples from stored responses.

        If responses have not been generated yet, this method calls
        :meth:`set_api_responses`. It then splits each stored response by newline
        and creates one row per line with columns ``'label'`` and ``'literal'``.

        Returns
        -------
        tuple: (pandas.dataframe, int)
            pandas.DataFrame
                DataFrame with one row per generated sentence, containing:
                - ``label``: label key from `label_contents_map_dict`
                - ``literal``: generated sentence (line)

        Side Effects
        ------------
        Updates `self.synth_data_df` with the generated DataFrame.
        """
        if label_col not in data_df.columns or text_col not in data_df.columns:
            raise ValueError(f"DataFrame must have columns '{label_col}' and '{text_col}'")

        pruned_df = data_df[[label_col, text_col]].copy()
        pruned_df = pruned_df.dropna(subset=[label_col, text_col])

        pruned_df[text_col] = pruned_df[text_col].astype(str).str.strip()
        pruned_df[label_col] = pruned_df[label_col].astype(str).str.strip()

        pruned_df = pruned_df[(pruned_df[text_col] != "") & (pruned_df[label_col] != "")]
        if pruned_df.empty:
            raise ValueError("No valid examples found in the DataFrame")
        return pruned_df, len(data_df)-len(pruned_df)

    @staticmethod
    def get_class_distribution(label_l):
        """
        Compute label frequency statistics for a classification dataset.

        Parameters
        ----------
        label_l : list
            List of labels (one per example).

        Returns
        -------
        examples_per_label : collections.Counter
            Counts per label.
        n_labels : int
            Number of unique labels.
        min_label : int
            Minimum examples among labels.
        max_label : int
            Maximum examples among labels.
        ratio_max_min : float or None
            Ratio ``max_label / min_label``. Returns None if `min_label` is 0.
        sub_rep : list
            Labels with counts below the 1st 25-quantile (under-represented).
        over_rep : list
            Labels with counts above the 24th 25-quantile (over-represented).

        Notes
        -----
        Quantiles are computed using ``statistics.quantiles(..., n=25)``.
        """
        examples_per_label = Counter(label_l)
        n_labels = len(examples_per_label)
        min_label = min(examples_per_label.values())
        max_label = max(examples_per_label.values())
        q_list = stats.quantiles(examples_per_label.values(), n=25)
        sub_rep = [key for key, value in examples_per_label.items() if value < q_list[0]]
        over_rep = [key for key, value in examples_per_label.items() if value > q_list[23]]
        ratio_max_min = max_label / min_label if min_label > 0 else None

        return examples_per_label, n_labels, min_label, max_label, ratio_max_min, sub_rep, over_rep

    @staticmethod
    def count_words(t: str) -> int:
        """
        Count whitespace-separated words in a string.

        Parameters
        ----------
        t : str
            Input text.

        Returns
        -------
        int
            Number of tokens obtained via ``t.split()``.
        """
        return len(t.split())

    @staticmethod
    def get_length_global_stats(text_l):
        """
        Compute global descriptive statistics of text length (in words).

        Parameters
        ----------
        text_l : list of str
            List of texts.

        Returns
        -------
        dict
            Dictionary with keys: ``min``, ``max``, ``mean``, ``q1``, ``q2``, ``q3``,
            and ``std_dev`` (population std).
        """
        lengths = [SyntDataGenerator.count_words(t) for t in text_l]
        quantiles = stats.quantiles(lengths, n=4)
        length_stats_global = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": stats.mean(lengths),
            "q1": quantiles[0],
            "q2": quantiles[1],
            "q3": quantiles[2],
            "std_dev": stats.pstdev(lengths) if len(lengths) > 1 else 0.0,
        }
        return length_stats_global

    @staticmethod
    def get_length_class_stats(text_l, label_l):
        """
        Compute per-label descriptive statistics of text length (in words).

        Parameters
        ----------
        text_l : list of str
            List of texts.
        label_l : list
            Labels aligned with `text_l`.

        Returns
        -------
        dict
            Mapping ``label -> stats`` where stats includes: ``n``, ``min``, ``max``,
            ``mean``, ``q1``, ``q2``, ``q3``, and ``std_dev``. Quantiles are None
            when a label has only one example.
        """
        lengths = [SyntDataGenerator.count_words(t) for t in text_l]
        lengths_per_label = defaultdict(list)
        for lab, length in zip(label_l, lengths):
            lengths_per_label[lab].append(length)
        length_stats_per_label = {}
        for lab, lenghts in lengths_per_label.items():
            if len(lenghts) > 1:
                quantiles = stats.quantiles(lenghts, n=4)
            else:
                quantiles = [None]*3
            length_stats_per_label[lab] = {
                "n": len(lenghts),
                "min": min(lenghts),
                "max": max(lenghts),
                "mean": stats.mean(lenghts),
                "q1": quantiles[0],
                "q2": quantiles[1],
                "q3": quantiles[2],
                "std_dev": stats.pstdev(lenghts) if len(lenghts) > 1 else 0.0,
            }
        return length_stats_per_label

    @staticmethod
    def get_tokens_stats(text_l, label_l, n_top_vocab):
        """
         Compute vocabulary statistics globally and per label.

         Texts are lowercased and tokenized with the regex pattern ``\\w+``.

         Parameters
         ----------
         text_l : list of str
             List of texts.
         label_l : list
             Labels aligned with `text_l`.
         n_top_vocab : int
             Number of top global tokens to return.

         Returns
         -------
         vocab_size : int
             Number of unique tokens in the full corpus.
         lexical_diversity : float
             Ratio ``vocab_size / n_tokens`` (0.0 if no tokens).
         top_vocab : list of tuple
             List of ``(token, count)`` pairs for the `n_top_vocab` most frequent tokens.
         vocab_per_label : dict
             Mapping ``label -> {'vocab_size': int, 'lexical_diversity': float}``.

         Notes
         -----
         Lexical diversity is computed as unique tokens divided by total tokens.
         """
        token_pattern = re.compile(r"\w+", re.UNICODE)
        all_tokens = []
        tokens_per_label = defaultdict(list)

        for lab, txt in zip(label_l, text_l):
            text_norm = txt.lower()
            tokens = token_pattern.findall(text_norm)
            all_tokens.extend(tokens)
            tokens_per_label[lab].extend(tokens)

        if all_tokens:
            vocab_counter = Counter(all_tokens)
            vocab_size = len(vocab_counter)
            lexical_diversity = vocab_size / len(all_tokens)
            top_vocab = vocab_counter.most_common(n_top_vocab)
        else:
            vocab_size = 0
            lexical_diversity = 0.0
            top_vocab = []

        vocab_per_label = {}
        for lab, toks in tokens_per_label.items():
            if toks:
                c = Counter(toks)
                vocab_per_label[lab] = {
                    "vocab_size": len(c),
                    "lexical_diversity": len(c) / len(toks),
                }
            else:
                vocab_per_label[lab] = {
                    "vocab_size": 0,
                    "lexical_diversity": 0.0,
                }
        return vocab_size, lexical_diversity, top_vocab, vocab_per_label

    @staticmethod
    def duplicated_text_stat(texts_l, labels_l, n_top_duplicates, n_examples):
        """
        Compute duplicate-text statistics for a labeled dataset.

        Texts are normalized using ``strip().lower()`` to detect duplicates.

        Parameters
        ----------
        texts_l : list of str
            Text examples.
        labels_l : list
            Labels aligned with `texts_l`.
        n_top_duplicates : int
            Maximum number of duplicated texts to include in the detailed output.
        n_examples : int
            Total number of examples (used to compute the repeated ratio).

        Returns
        -------
        n_duplicated_texts : int
            Number of *distinct* normalized texts that appear more than once.
        n_examples_in_duplicated_texts : int
            Total number of examples that belong to duplicated texts (counts all occurrences).
        n_repeated_examples : int
            Number of repeated examples beyond the first occurrence for each duplicated text.
        ratio_repeated_examples : float
            Ratio ``n_repeated_examples / n_examples``.
        top_duplicates : list of dict
            List of dictionaries with keys:
            - ``text_norm``: normalized duplicated text
            - ``frequency``: occurrence count
            - ``labels``: sorted list of labels where it appears
        """
        texts_norm = [t.strip().lower() for t in texts_l]
        text_freq = Counter(texts_norm)

        n_duplicated_texts = sum(1 for _, c in text_freq.items() if c > 1)
        n_examples_in_duplicated_texts = sum(c for c in text_freq.values() if c > 1)
        n_repeated_examples = n_examples_in_duplicated_texts - n_duplicated_texts
        ratio_repeated_examples = n_repeated_examples / n_examples

        # For each normalized text, which labels does it appear in?
        labels_per_text = defaultdict(set)
        for lab, txt in zip(labels_l, texts_l):
            key = txt.strip().lower()
            labels_per_text[key].add(lab)

        top_duplicates = []
        for text_n, c in text_freq.most_common():
            if c <= 1:
                break
            top_duplicates.append(
                {
                    "text_norm": text_n,
                    "frequency": c,
                    "labels": sorted(list(labels_per_text[text_n])),
                }
            )
            if len(top_duplicates) >= n_top_duplicates:
                break
        return n_duplicated_texts, n_examples_in_duplicated_texts,  n_repeated_examples, ratio_repeated_examples, top_duplicates

    @staticmethod
    def get_meta_analysis_data_df(
        df,
        label_col="label",
        text_col="literal",
        n_top_vocab=30,
        n_top_duplicates=20,
    ):
        """
        Run a meta-analysis on a labeled text dataset.

        This utility computes dataset-level statistics (class distribution),
        length statistics (global and per-label), vocabulary statistics (global and
        per-label), and duplicate-text statistics.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing at least `label_col` and `text_col`.
        label_col : str, default='label'
            Column name containing labels.
        text_col : str, default='literal'
            Column name containing text samples.
        n_top_vocab : int, default=30
            Number of most frequent tokens to return in the global vocabulary summary.
        n_top_duplicates : int, default=20
            Maximum number of duplicated texts to include in the detailed duplicates list.

        Returns
        -------
        summary : dict
            Nested dictionary with the following top-level keys:
            - ``dataset``: counts per label and imbalance indicators
            - ``lengths``: global and per-label length stats
            - ``vocabulary``: global and per-label vocab stats
            - ``duplicates``: duplicate-text stats and examples

        Raises
        ------
        ValueError
            If the DataFrame does not contain the required columns or becomes empty
            after cleaning.

        See Also
        --------
        clean_data_df, get_class_distribution, get_length_global_stats,
        get_length_class_stats, get_tokens_stats, duplicated_text_stat
        """
        df, n_empty_cols = SyntDataGenerator.clean_data_df(df, label_col, text_col)
        labels = df[label_col].tolist()
        texts = df[text_col].tolist()
        n_examples = len(texts)
        examples_per_label, n_labels, min_label, max_label, ratio_max_min, sub_rep, over_rep = SyntDataGenerator.get_class_distribution(labels)
        length_stats_global = SyntDataGenerator.get_length_global_stats(texts)
        length_stats_per_label = SyntDataGenerator.get_length_class_stats(texts, labels)
        vocab_size, lexical_diversity, top_vocab, vocab_per_label = SyntDataGenerator.get_tokens_stats(
            texts,
            labels,
            n_top_vocab
        )
        n_duplicated_texts, n_examples_in_duplicated_texts,  n_repeated_examples, ratio_repeated_examples, top_duplicates = SyntDataGenerator.duplicated_text_stat(
            texts,
            labels,
            n_top_duplicates,
            n_examples
        )
        summary = {
            "dataset": {
                "n_empy_text": n_empty_cols,
                "n_examples": n_examples,
                "n_labels": n_labels,
                "examples_per_label": dict(examples_per_label),
                'n_max_label': max_label,
                'n_min_label': min_label,
                "ratio_max_min_label": ratio_max_min,
                "classes_under_5%_data": sub_rep,
                "classes_over_95%_data": over_rep
            },
            "lengths": {
                "global": length_stats_global,
                "per_label": length_stats_per_label,
            },
            "vocabulary": {
                "vocab_size_global": vocab_size,
                "lexical_diversity_global": lexical_diversity,
                "top_vocab_global": top_vocab,
                "per_label": vocab_per_label,
            },
            "duplicates": {
                "n_duplicated_texts": n_duplicated_texts,
                "n_examples_in_duplicated_texts": n_examples_in_duplicated_texts,
                "n_repeated_examples": n_repeated_examples,
                "ratio_repeated_examples": ratio_repeated_examples,
                "top_duplicates": top_duplicates,
            },
        }
        return summary


class OnyxiaSyntDataGenerator(SyntDataGenerator):

    def api_call(self, prompt):
        """
        Call the Onyxia LLM chat completion endpoint.

        Parameters
        ----------
        prompt : str
            User prompt to send to the remote chat-completions API.

        Returns
        -------
        str
            The model reply content extracted from the JSON response.

        Raises
        ------
        requests.RequestException
            If the HTTP request fails.
        KeyError
            If the expected JSON fields are missing in the response payload.
        """
        url = "https://llm.lab.sspcloud.fr/api/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
          "model": f"{self.model}",
          "messages": [
            {
              "role": "user",
              "content": prompt
            }
          ]
        }
        response = requests.post(url, headers=headers, json=data)
        reply = response.json()["choices"][0]["message"]["content"]
        return reply


class GroqSyntDataGenerator(SyntDataGenerator):
    """
    Initialize a Groq-based synthetic data maker with multiple API keys.

    This subclass extends :class:`SyntDataGenerator` by accepting a list of API keys
    (`api_key_l`) that can be rotated when rate limits are reached.

    Parameters
    ----------
    root, model, label_contents_map_dict, body_intro, task, n_responses, p_lang, o_lang, output_format, long, eg_filename, eg_strategy
        See :meth:`SyntDataGenerator.__init__`.
    api_key_l : list of str
        List of Groq API keys used for key rotation.
    api_key : str or None, optional
        Initial key passed to the parent. The actual key used may later be
        replaced by elements from `api_key_l`.

    Returns
    -------
    None
    """
    def __init__(
        self,
        root,
        api_key_l,
        model,
        label_contents_map_dict,
        body_intro,
        api_key=None,
        task=None,
        n_responses=None,
        p_lang='en',
        o_lang=None,
        output_format=None,
        len_constraint=None,
        eg_filename='static_examples.json',
        eg_strategy='static'
    ):
        super().__init__(
            root=root,
            model=model,
            label_contents_map_dict=label_contents_map_dict,
            body_intro=body_intro,
            api_key=api_key,
            task=task,
            n_responses=n_responses,
            p_lang=p_lang,
            o_lang=o_lang,
            output_format=output_format,
            len_constraint=len_constraint,
            eg_filename=eg_filename,
            eg_strategy=eg_strategy
        )
        self.api_key_l = api_key_l

    def api_call(self, prompt):
        """
        Call the Groq chat completion API.

        Parameters
        ----------
        prompt : str
            User prompt to send to Groq.

        Returns
        -------
        str
            Model-generated text, stripped of leading/trailing whitespace.

        Raises
        ------
        Exception
            Propagates exceptions raised by the Groq client (e.g., authentication,
            connectivity, or rate limiting).
        """
        client = Groq(api_key=self.api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.4
        )
        return chat_completion.choices[0].message.content.strip()

    def set_api_responses(self):
        """
        Generate prompts and store Groq responses with API-key rotation on rate limits.

        This method behaves like :meth:`SyntDataGenerator.set_api_responses`, but if a
        ``RateLimitError`` with status code 429 is raised, it switches to the next
        API key from `self.api_key_l` and retries.

        Returns
        -------
        None
            Responses are stored in-place in `self.label_contents_map_dict`.

        Raises
        ------
        RuntimeError
            If all API keys in `api_key_l` have reached their limits.
        RateLimitError
            If a rate-limit error occurs that is not handled (e.g., different status code).
        """
        self.prompt_maker.logger.info('Setting api responses...')
        x = 0
        self.api_key = self.api_key_l[x]
        for data_d in tqdm(self.label_contents_map_dict.values(), total=len(self.label_contents_map_dict.values())):
            self.prompt_fields['description'] = data_d.get('description')
            if self.eg_strategy:
                self.prompt_fields['example'] = data_d.get('example')
            prompt = self.prompt_maker.get_prompt(self.prompt_fields)
            data_d['prompt'] = prompt
            while True:
                try:
                    data_d['response'] = self.api_call(prompt)
                    break
                except RateLimitError as e:
                    if e.status_code == 429:
                        print(f"Key {x} limit reached, switching key...")
                        x += 1
                        self.api_key = self.api_key_l[x]
                        if x >= len(self.api_key_l):
                            raise RuntimeError("All API keys have reached their limits.")


class AzureSyntDataGenerator(SyntDataGenerator):

    def api_call(self, prompt):
        """
        Call an Azure OpenAI chat completion deployment.

        Parameters
        ----------
        prompt : str
            User prompt to send to the Azure OpenAI chat completion endpoint.

        Returns
        -------
        str
            The model reply content.

        Raises
        ------
        Exception
            Propagates exceptions raised by the Azure OpenAI client (e.g.,
            authentication, connectivity, quota/rate limits).
        """
        endpoint = "https://open-ai-ine.cognitiveservices.azure.com/"
        deployment = self.model

        subscription_key = self.api_key
        api_version = "2024-12-01-preview"

        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in generation of sentences for NLP model trainning",
                },
                {
                    "role": "user",
                    "content": f"{prompt}",
                }
            ],
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            model=deployment
        )
        return response.choices[0].message.content


