![](https://www.ine.es/menus/_b/img/logoINESocial.png)

# codauto v.0.1.0

## Description
The idea of this library is to assist official statisticians in classifying variables according to standards, 
such as economic activities using [CNAE](https://ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736177032&menu=ultiDatos&idp=1254735976614), 
the standard classification for economic activities in Spanish official statistics (the Spanish version of [NACE](https://ec.europa.eu/eurostat/web/nace)).

This library is designed to train, evaluate, and use any model for any standard.

It has two modes for prediction: codification and assistance. Codification returns a code, while assistance returns a list of the most likely codes. It can also be used to classify a large list of data.

## Requirements
**C ++ compiler** for install fasttext if you are working on windows.

**fasttext v0.9.2** .

**python v3.9.18**.

You can use *entorno_codpython_20240613.yml* to easily install **codauto** in a standardized conda environment.

## Usage
Here you will find a brief explanation of how to install and use codauto in order to classify economic activities with CNAE 09 or 25 in a conda env using Windows 10.
### Installation
#### Generate Python env with Conda (optional)
If you work in windows and have installed Microsoft Visual Studio tools, follow this instructions to create an environment using *entorno_codpython_20240613.yml*  in Conda:

1. **Set the path to C++ compiler**
You must find the directory where cl compiler is located. Then, in the Conda Prompt, type:
````conda
set PATH=%PATH%;{compiler path}
````
where {compiler path} is the path to your cl directory.

2. **Change the environment name (optional)**
The default name is *entorno_codpython_20240613*. To change it, open *entorno_codpython_20240613.yml*  and modify the name in the NAME field.

3. **Create the environment**
In the Conda Prompt, navigate to the directory where *entorno_codpython_20240613.yml* is located using *cd* and type:
````conda
conda env create -f entorno_codpython_20240613.yml
````
### Generate a Structured child for your classification.
This library provides an abstract class for processing any standard structure, *Structured*. 
For any standard, you must code a *Structured* child that implements its own method *get_level(code)*. 
This method's input must be a code as a string, and its output must be an integer indicating the hierarchical level of the code, with 0 indicating the highest level.

An example for CNAE-2025.
```python
from codauto import Structured

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

structurecnae25 = StructuredCNAE(
    structure_df=structure_df,
    names_l=['section', 'division', 'group', 'class']
)
```
Parameters for any *Structured*:
- *structure_df* (pandas.DataFrame): Must contain two columns:
    - Column 0 (str): CNAE hierarchical sorted codes (possibly with dot notation).
    - Column 1 (str): Category titles corresponding to each code.
- *names_l* (None|tuple(str), Optional): Names of the different hierarchy levels. If it is None, it will be generated automatically.

### Generate a Codifier child for your model.
This library provides an abstract class that can be used to implement any classification strategy that provides confidence values for each possible code in the standard, *Codifier*. 
For any strategy the child class must include at least 4 methods:
- *train(**kgrams)*: This is an abstract method designed to receive a dictionary of parameters, which allows it to be easily configured for training models with different NLP libraries available in Python (fastText, BERT, etc.). Use self.model.
- *load(name)*: This is an abstract method intended to implement a loading procedure for a previously trained model with the name {name} in self.model.
- *save(name)*: This is an abstract method intended to implement a saving procedure for a previously trained model with the name {name} from self.model.
- *get_preds_for_batch(sample, idxs, preprocess)*:This is an abstract method intended to generate predictions on a batch of input samples.
It takes as parameters a list of texts (samples) to be predicted or classified, a list of indices (idxs) that identify each sample in the batch, 
and an optional flag (clean_samples) that allows applying cleaning or preprocessing before prediction. 
The method must return a dictionary where each key corresponds to an index from idxs, and each value is itself a dictionary with all possible labels ordered by confidence, 
along with the associated confidence scores for each hierarchical level. 

For example, suppose the classification consists of two hierarchical levels: 
two Sections (A, B), each with two Groups (A1, A2) and (B1, B2), respectively.
For each sample included (in this example, two samples with indices 0 and 1), the method should return something like:
```python
{ 
 0: { 
    "label_section": ["A", "B"],
    "score_section": [0.85, 0.15], 
    "label_group": ["A1", "A2", "B1", "B2"],
    "score_group": [0.55, 0.30, 0.1, 0.05]
    }, 
 1: { 
    "label_section": ["B", "A"], 
    "score_section": [0.57, 0.43],
    "label_group": ["B1", "A1", "B2", "A2"], 
    "score_group": [0.5, 0.4, 0.07, 0.03] 
    }
} 

```
The main method that conect the child with *Codifier* is *get_preds_for_batch(samples, idxs, preprocess)*. 
The other methods can either be fully implemented in the child class or simply defined with pass if no logic is needed.

*Codifier* has this parameters:
- *structure_instance*(Structured child): Structured child for the standard
- *train_df* (pandas.DataFrame): Must contain at least two columns:
    - Column 0 (str): Codes (possibly with dot notation).
    - Column 1 (str): Text description.
- *test_df* (pandas.DataFrame): Must contain three columns:
    - Column 0 (str): Ground truth last level code.
    - Column 1 (str): Text descriptions.
    - Column 2 (str): The source of the description, it can be empty but must be.
- *root_path* (str): Directory path where models will be save and load. If the directory does not exit it will be created.
- *corres_df* (pandas.DataFrame|None): Optional must contain three columns, defaults value None:
    - Column 0 (str): The previous classification code.
    - Column 1 (str): The new classification code.
    - Column 2 (int): The hierarchical level of the correspondence.
- *min_length_texts* (int): Minimum number of characters for text descriptions. Defaults to 3

## CodifierFastText
The codauto library already provides an implemented child class that allows supervised training of models with the fastText library that classify using a bottom-up strategy. 
The help command can be used to access the documentation and start training directly.

```python
# generate a CodifierFastText instance for CNAE
from codifier import CodifierFastText

cnae_structure = StructuredCNAE(
    structure_df,
    ['section', 'division', 'group', 'class']
)

cnae_codifier = CodifierFastText(
    root_path=root_path,
    structure_instance=cnae_structure,
    train_df=train_df,
    test_df=test_df,
    corres_df=corres_df
)
```
### Train
This method prepares the training data in FastText format, optionally uses
pre-trained word vectors, and trains a supervised model. Training duration
is logged.

Parameters:
- **kwargs** (*dict*): Optional keyword arguments: 
    - *epoch* (*int*, default=10): Number of training epochs.
    - *lr* (*float*, default=0.1): Learning rate for training.
    - *wordNgrams* (*int*, default=3): Maximum length of word n-grams.
    - *pretrained_vectors* (*str* or *None*, default=None): Filename of pre-trained word vectors to use.  
    - *train_set* (*pandas.DataFrame*, default=self.train_df): Training dataset with labels in the first column and text in the second.

### Save
The save method saves the trained model in *self.root_path/<name>*.
```python
cnae_codifier.save(name)
```
Parameters:
- *name* (str): Filename to use when saving the model.
### Load
The load method loads the trained model from *self.root_path/<name>*.
```python
cnae_codifier.load(name)
```
Parameters:
- *name* (str): Filename of the saved FastText model.
### Evaluate
The evaluate method evaluates the performance of the model using the test_df provided. It calculates various metrics and optionally includes a precision-recall curve.
```python
# If simplify is True
metrics = cnaecod.evaluate(
    source='all',
    get_curve=True,
    simplify=True,
    version='CNAEmodel'
)
#Else
metrics, report = cnaecod.evaluate(
    source='all',
    get_curve=True,
    simplify=False,
    version='CNAEmodel'
)
```
Parameters:
- *source* (str, optional): Source of the test data to evaluate. If 'all' (default), uses the entire test set. Otherwise, filters the test set by the specified source value.
- *get_curve* (bool, optional): Whether to plot a precision-recall curve. Defaults value True
- *simplify* (bool, optional): If True, return basic metrics; if False,return detailed metrics and a class metric report. Defaults value True
- *version* (str, optional): Version of the CNAE model to evaluate. Defaults value CNAEmodel

Returns:
- *metrics* (dict): A dictionary of evaluation metrics per level. Includes accuracy and F1 scores (micro, macro, weighted).
- *report* (dict, optional):A dictionary of classification reports per label and level. Only returned if `simplify` is False.
### Predict
The predict method can provide different hierarchical level codes and also return a list of the most likely codes for your economic activity.
```python
  # Set parameters
desc_l = [
    'compra y venta de vehículos de motor'
    ]
identifier_l = ['A01']
mode = 'codification'
hierarchical_level = 1
original_code_l0[G]
threshold = 0.01

prediction = cnaecod09.predict(
    desc_l=desc_l,
    mode=mode,
    hierarchical_level=hierarchical_level,
    threshold=threshold,
    original_code_l=None,
    identifier_l=identifier_l
)
````
Parameters:
- *desc_l* (tuple(str)): List of textual descriptions to classify.
- *mode* (str): Prediction mode. Must be either:
    - "codification" : return only top-1 prediction.
    - "assistance"   : return up to 15 predictions above threshold.
- *hierarchical_level* (int): The hierarchical classification level (1-based index).
- *threshold* (float): Minimum confidence required for a prediction to be considered valid.
- *original_code_l*  (tuple(str|None)|None, optional):List of previously known codes for each sample (for recoding assistance).If provided, corres_df must be defined. 1:1 correspondence will be directly recoded. Defaults None
- *identifier_l* (tuple(any), optional):List of identifiers corresponding to each input sample (e.g., for traceability). If its None will be made automatically. Defaults None

Returns:
The output stored in prediction is a sorted list of dictionaries (dict), where each dict from each dictionary corresponds to an activity and is in the same list position as the activity in the description_l list. The dictionaries are structured as follows:
```python
[{
'description': 'compra y venta de vehículos de motor',
'original_code': 'G',
'label': ('G',),
'confidence': (100,),
'hierarchical_level': 1,
'identifier': 'A01',
'title': ('Comercio al por mayor y al por menor; reparación de vehículos de motor y motocicletas', )
}]
```
- 'description': input description.
- 'original_code': original code used for recoding (if any).
- 'label': tuple of predicted class codes.- 'confidence': tuple of confidence scores (integers, 0-100).
- 'confidence': tuple of confidence scores (integers, 0-100).
- 'hierarchical_level': classification level used.
- 'identifier': identifier from input or generated.- 'title': tuple of category titles corresponding to the labels.
- 'title': tuple of category titles corresponding to the labels.

## Prompt generation
This library also has two functions for generating prompts to produce synthetic data with LLMs via two different approaches using explanatory notes.

#### Fill_prompt_synt_data
It fills a synthetic prompt template by replacing predefined placeholders with a given title and notes. The purpose of this function is to generate prompts for requesting that LLMs directly generate samples of coded descriptions.
```python
from codauto.prompt_maker import fill_prompt_synt_data

fill_prompt_synt_data(title, includes, prompt)
```
Parameters:
- *title* (str): The title to insert into the prompt.
- *includes* (str): The includes explanatory notes to insert into the prompt.
- *prompt* (str): Prompt template

#### Fill_prompt_aug_data
It fills a synthetic prompt template by replacing predefined placeholders with a given title and notes. 
The purpose of this function is to generate prompts for requesting that LLMs create dictionaries of synonyms from the keywords in the title, and then generate samples of coded descriptions by replacing the words in the title.
```python
from codauto.prompt_maker import fill_aug_synt_data

fill_prompt_aug_data(title, includes, language, prompt)
```
Parameters:
- *title* (str): The input title to extract keywords from.
- *includes* (str): The includes explanatory notes to insert into the prompt.
- *language* (str, optional): The language to use for stopword filtering. Defaults value 'spanish'.
- *prompt* (str): Prompt template

## References
Install [Visual studio C++ compiler.](https://github.com/bycloudai/InstallVSBuildToolsWindows)

[Fasttext library.](https://fasttext.cc/docs/en/support.html)

[Create conda env.](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

What are [.yml files.](https://en.wikipedia.org/wiki/YAML)

## Support
Write an email to any author asking for help.

You can also contact via nomenclaturas@ine.es.

## Contributing
This is a private project, but any ideas are welcome. You can email any author to contribute.

## Authors
Stadistics Spain (git.metodologia@ine.es) 

## Acknowledgment
Sebastián Gallego Herrera (sebastian.gallego.herrera@ine.es)

Andrés Jurado Prieto (andres.jurado.prieto@ine.es)

Adrián Pérez Bote (adrian.perez.bote@ine.es)

Jorge Fernández Calatrava (jorge.fernandez.calatrava@ine.es)
