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
Created on Tue Sep  9 13:40:12 2025

@author: U853768
"""
import numpy as np
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline,
)
from datasets import Dataset, ClassLabel, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

from statcodgen.codifier import Codifier


class CodifierBERT(Codifier):
    """
    CodifierBERT is a BERT-based text classification codifier built on the Hugging Face
    Transformers ecosystem. It extends the `Codifier` base class to support model
    training, evaluation, inference, and persistence for hierarchical or structured
    text classification tasks.

    The class automates dataset preparation, tokenizer setup, metric computation,
    model fine-tuning, and prediction generation.

    Parameters
    ----------
    structure_instance : object
        Structured child instance representing label hierarchy or taxonomy information.
    train_df : pandas.DataFrame
        Training dataset containing two columns, first must be the label and second the text.
    test_df : pandas.DataFrame
        Test dataset for evaluation purposes. It must have 3 columns: first column must be label, second text and third source (source can be empty)
    root_path : str
        Path to the directory where models, logs, and checkpoints are stored.
    model_id : str
        Hugging Face model identifier (e.g., 'bert-base-uncased').
    corres_df : pandas.DataFrame, optional
        Correspondence dataframe linking labels or hierarchical mappings with other classifications.
    min_lenght_texts : int, default=3
        Minimum number of tokens required for a text to be considered valid.
    preprocess_text : callable function or None Default Value: None.
        If a text preprocessing function is given, applies such function. 
        If None, does not apply text preprocessing.
    temperature : float or None, default=None
        Temperature scaling factor for calibrating model confidence. A value
        greater than 1 softens the output distribution (fixes overconfidence),
        while a value below 1 sharpens it (fixes underconfidence). If None,
        no temperature scaling is applied and the standard pipeline is used
        for inference
  

    Attributes
    ----------
    tokenizer : AutoTokenizer
        Tokenizer loaded from Hugging Face using `model_id`.
    label_encoder : LabelEncoder
        Scikit-learn label encoder for mapping textual labels to integers.
    id2label : dict of int to str
        Mapping from integer class indices to label strings.
    label2id : dict of str to int
        Reverse mapping from label strings to class indices.
    class_label : datasets.ClassLabel
        Hugging Face datasets label handler for encoding.
    device : torch.device
        Active computing device ('cuda' or 'cpu').
    train_data : datasets.DatasetDict
        Processed Hugging Face dataset for training and validation.
    classifier : transformers.Pipeline
        Inference pipeline for batched predictions.
    temperature : float or None
        Temperature scaling factor. See parameter description above.
    """

    def __init__(self,
                 structure_instance,
                 root_path,
                 model_id,
                 train_df=None,
                 test_df=None,
                 corres_df=None,
                 min_lenght_texts=3,
                 device='cuda',
                 preprocess=None,
                 temperature=None
                 ):
        super().__init__(
            structure_instance=structure_instance,
            train_df=train_df,
            test_df=test_df,
            root_path=root_path,
            corres_df=corres_df,
            min_lenght_texts=min_lenght_texts,
            preprocess=preprocess,
        )
        local_path = os.path.abspath(
            os.path.join(self.root_path, model_id)
        )
        if os.path.exists(local_path):
            self.model_id = local_path
        else:
            self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(self.structure.reversed_hierarchy.keys()))
        self.id2label = {i: str(label) for i, label in enumerate(self.label_encoder.classes_)}
        self.label2id = {label: i for i, label in self.id2label.items()}
        self.class_label = ClassLabel(names=self.label_encoder.classes_.tolist())
        self.device = device
        self.train_data = None
        self.classifier = None
        self.temperature = temperature

    def tokenize(self, example, max_seq_len=256):
        """
        Tokenizes input text examples for model ingestion.

        Parameters
        ----------
        example : dict
            Input example containing the 'target' text field.
        max_seq_len : int, default=256
            Maximum sequence length for padding and truncation.

        Returns
        -------
        dict
            Tokenized representation of the input suitable for model processing.
        """
        return self.tokenizer(
            example['target'],
            padding='max_length',
            truncation=True,
            max_length=max_seq_len
        )

    def get_pipeline(self):
        """
        Initializes a Hugging Face text-classification pipeline for inference.

        Notes
        -----
        Automatically selects the appropriate device (GPU or CPU) based on availability.
        """
        device = 0
        if self.device == 'cpu':
            device = -1
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,
            device=device
        )

    @staticmethod
    def compute_metrics(eval_pred):
        """
        Computes performance metrics for model evaluation.

        Parameters
        ----------
        eval_pred : tuple
            A tuple containing (logits, labels) from the evaluation step.

        Returns
        -------
        dict
            Dictionary containing the following metrics:
            - 'accuracy': float
            - 'f1_macro': float
            - 'precision_macro': float
            - 'recall_macro': float
            - 'top_1_accuracy': float
            - 'top_2_accuracy': float
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        num_classes = logits.shape[1]
        class_labels = np.arange(num_classes)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)
        top_1_acc = top_k_accuracy_score(labels, logits, k=1, labels=class_labels)
        top_2_acc = top_k_accuracy_score(labels, logits, k=2, labels=class_labels)
        return {
            'accuracy': accuracy,
            'f1_macro': f1,
            'precision_macro': precision,
            'recall_macro': recall,
            'top_1_accuracy': top_1_acc,
            'top_2_accuracy': top_2_acc,
        }

    def set_train_data(self, batched, test_size, seed):
        """
        Prepares and encodes training and validation datasets using Hugging Face Datasets.

        Parameters
        ----------
        batched : bool
            Whether to apply transformations in batched mode.
        test_size : float
            Fraction of data reserved for validation.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        None
            The prepared dataset is stored in `self.train_data`.
        """
        train_df = self.train_df.copy()
        train_df.columns = ['label', 'target']
        data = DatasetDict({
            'train': Dataset.from_pandas(train_df[['label', 'target']])
        })
        data = data.map(
            lambda x: {'label': self.label_encoder.transform(x['label'])}, batched=batched
        )
        data = data.cast_column('label', self.class_label)
        data = data['train'].train_test_split(
            test_size=test_size, seed=seed#, stratify_by_column='label'
        )
        data["validation"] = data.pop("test")
        self.train_data = data

    def train(self, train_args=None, batched=True, test_size=0.05, seed=42):
        """
        Fine-tunes the BERT model using Hugging Face's Trainer API.

        This method configures and launches fine-tuning for the underlying
        transformer model (e.g., BERT, RoBERTa, DistilBERT) on a labeled dataset
        for sequence classification tasks. It uses the Hugging Face `Trainer`
        and `TrainingArguments` APIs for efficient and configurable training.

        Parameters
        ----------
        train_args : dict, optional
            Additional or overriding keyword arguments for `TrainingArguments`.
            Common keys include:
                - output_dir : str
                    Directory to save logs and model outputs.
                - num_train_epochs : int
                    Number of training epochs. Default is 2.
                - learning_rate : float
                    Learning rate for the optimizer. Default is 5e-5.
                - lr_scheduler_type : str
                    Type of learning rate scheduler (e.g., 'linear', 'cosine').
                - per_device_train_batch_size : int
                    Training batch size per device. Default is 16.
                - per_device_eval_batch_size : int
                    Evaluation batch size per device. Default is 32.
                - gradient_accumulation_steps : int
                    Number of update steps to accumulate before performing a backward/update pass.
                - warmup_ratio : float
                    Fraction of total steps used for learning rate warmup.
                - weight_decay : float
                    Weight decay (L2 regularization) coefficient.
                - logging_dir : str
                    Directory for training logs.
                - logging_strategy : str
                    When to log metrics ('epoch', 'steps', etc.). Default is 'epoch'.
                - evaluation_strategy : str
                    When to run evaluation ('epoch', 'steps', etc.). Default is 'epoch'.
                - save_strategy : str
                    Model checkpoint saving strategy. Default is 'no'.
                - fp16 : bool
                    Whether to use mixed precision (float16) training. Default is False.
                - load_best_model_at_end : bool
                    Whether to reload the best model after training. Default is False.
                - report_to : list
                    Integrations for logging/reporting (e.g., ['tensorboard', 'wandb']).
                - log_level : str
                    Logging verbosity level. Default is 'warning'.

        batched : bool, default=True
            Whether tokenization should be applied in batches during dataset mapping.

        test_size : float, default=0.05
            Fraction of the dataset reserved for validation during training.

        seed : int, default=42
            Random seed used for reproducibility during dataset splitting.

        Returns
        -------
        None
            Trains and initializes the classification pipeline.
        """
        if train_args is None:
            train_args = {}
        args = dict(
            output_dir=self.root_path,
            num_train_epochs=2,
            learning_rate=5e-5,
            lr_scheduler_type='linear',
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_strategy='epoch',
            evaluation_strategy='epoch',
            save_strategy='no',
            fp16=False,
            load_best_model_at_end=False,
            report_to=[],
            log_level='warning',
            max_seq_len = 256,
            local_rank = -1
        )
        args.update(train_args)
        if self.train_data is None:
            self.set_train_data(batched, test_size, seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
        ).to(self.device)
        tokenized_data = self.train_data.map(lambda batch: self.tokenize(batch, max_seq_len=args['max_seq_len']), batched=batched)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        args_to_train = args.copy()
        args_to_train.pop('max_seq_len', None)
        training_args = TrainingArguments(**args_to_train)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data['train'],
            eval_dataset=tokenized_data['validation'],
            compute_metrics=self.compute_metrics,
            data_collator=data_collator
        )
        trainer.train()
        self.model = trainer.model
        self.get_pipeline()

    def save(self, name='bert_model', save_tok=True):
        """
        Saves the trained model and optionally its tokenizer.

        Parameters
        ----------
        name : str, default='bert_model'
            Directory name for saving model.
        save_tok : bool, default=True
            Whether to save the tokenizer.

        Returns
        -------
        None
        """
        path = os.path.join(self.root_path, name)
        if save_tok:
            self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

    def load(self, name, load_tok=True):
        """
        Loads a previously trained model and tokenizer from disk.

        Parameters
        ----------
        name : str
            Name of the saved model directory.
        load_tok : bool, default=True
            Whether to load the tokenizer.

        Returns
        -------
        None
            Loads model and initializes the pipeline.
        """
        path = os.path.join(self.root_path, name)
        if load_tok:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.get_pipeline()

    def get_logits(self, samples, batch_size=16):
        """
        Computes raw logits for a list of text samples using the loaded model.

        Processes samples in mini-batches to avoid GPU memory overflow, moving
        each batch to the model's device and immediately releasing it after
        the forward pass.

        Parameters
        ----------
        samples : list of str
            Input text samples to classify.
        batch_size : int, default=16
            Number of samples processed per forward pass. Reduce if GPU
            memory is limited, increase for faster inference.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_samples, n_classes) containing the raw
            unnormalized logits for each sample. Returned on CPU.

        Notes
        -----
        Padding is applied within each mini-batch, so sequences are only
        padded to the longest sample in that batch rather than across the
        full input, reducing memory usage compared to full-batch inference.
        GPU memory is explicitly freed after each batch via torch.cuda.empty_cache().
        """
        model_device = next(self.model.parameters()).device
        all_logits = []
        self.model.eval()
        # It made by batch because of gpu memory
        for i in range(0, len(samples), batch_size):
            batch = samples[i: i + batch_size]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                all_logits.append(outputs.logits.cpu())
            del inputs, outputs
            torch.cuda.empty_cache()

        return torch.cat(all_logits, dim=0)

    def predict_proba_calibrated(self, samples):
        """
        Computes temperature-scaled class probabilities for a list of text samples.

        Applies temperature scaling to the raw logits before the softmax, which
        calibrates overconfident or underconfident models by softening or sharpening
        the output distribution. A temperature greater than 1 flattens probabilities
        toward a uniform distribution (fixes overconfidence), while a temperature
        below 1 sharpens them toward 0 or 1 (fixes underconfidence).

        Parameters
        ----------
        samples : list of str
            Input text samples to classify.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) containing the calibrated
            class probabilities for each sample. Values are in [0, 1] and
            sum to 1 along axis=1.

        Notes
        -----
        Requires self.temperature to be set (float, > 0) before calling.
        For uncalibrated inference use get_pred_for_batch() with temperature=None.
        Internally calls get_logits(), so batching and device management
        follow that method's behavior.
        """
        logits = self.get_logits(samples)

        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)

        return probs.cpu().numpy()

    def get_pred_for_batch(self, samples):
        """
        Generate predictions for a batch of text samples using Hugging Face pipeline.

        Parameters
        ----------
        samples : list of str
            Input text samples to classify.

        Returns
        -------
        list of tuple
            Each element contains:
            - labels (list of str): predicted labels sorted by confidence.
            - confidences (list of float): corresponding probabilities.
        """
        formatted = [None] * len(samples)
        if self.temperature is None:
            results = self.classifier(samples, truncation=True)
            for index, sample_result in enumerate(results):
                labels = [item["label"] for item in sample_result]
                confidences = [item["score"] for item in sample_result]
                formatted[index] = (labels, confidences)
        else:
            results = self.predict_proba_calibrated(samples)
            id2label = self.model.config.id2label
            for index, p in enumerate(results):
                pairs = [(id2label[i], float(p[i])) for i in range(len(p))]
                pairs.sort(key=lambda x: x[1], reverse=True)
                labels_sorted = [x[0] for x in pairs]
                probs_sorted = [x[1] for x in pairs]
                formatted[index] = (labels_sorted, probs_sorted)
        return formatted
