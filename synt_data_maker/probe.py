# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 13:25:01 2025

@author: U853768
"""
from process_notes import ProcessNACENB
import pandas as pd
import os
from synt_data_maker import OnyxiaSyntDataGenerator

notes_n = "notas_explicativas_noruego.csv"
train_n = "train_norwaydata.csv"
index_n = "NACE Rev. 2.1 - Index entries - Overview_translated_to_NB_with_deepl_complete.csv"
path = r'C:\Users\U000000\Documents\metodologia y de muestras\Programacion\Generador_de_base_de_datos\Noruego\input'
##########
notes_df = pd.read_csv(
    os.path.join(path, notes_n),
    dtype='str',
    encoding='latin-1',
    sep=';',
)
train_df = pd.read_csv(
    os.path.join(path, train_n),
    dtype='str'
)
index_df = pd.read_csv(
    os.path.join(path, index_n),
    dtype='str'
)
# Se obtiene el notes_dict #######
process_nb = ProcessNACENB(
    notes_df=notes_df[notes_df['level'] == "4"],
    examples_df=index_df,
    sample=5
)

process_nb.set_descriptions(
    col_c='code',
    col_d_l=['name', 'notes']
)

process_nb.set_key_words(
    col_c='CODE',
    col_k='KEYWORD_NB'
)

process_nb.example_df = train_df
process_nb.set_examples(
    col_c='nace_21_4digit_code',
    col_d='text_for_coding'
)
# Se generan datos sintéticos
body_intro = 'You are an expert Norwegian economist.'
data_dict = process_nb.get_dict_data()
data_maker = OnyxiaSyntDataGenerator(
        root=r'C:\Users\U000000\Documents\metodologia y de muestras\Programacion\Gitlab_Projects\statcodgen\synt_data_maker',
        api_key="sk-50116763c294421a8e796d17e9fab0bb",
        model="gpt-oss:120b",
        label_contents_map_dict=data_dict,
        body_intro=body_intro,
        task='generate',
        n_responses='11',
        p_lang='en',
        o_lang='sp',
        output_format='lines',
        len_constraint='short',
        eg_filename='static_examples.json',
        eg_strategy='static'
 )
response_df = data_maker.get_synt_data_df()
#meta_data_dict = data_maker.get_meta_analysis_data_df(reponse_df)
#data_dict_rep = data_maker.data_dict


#####_Se_generan_los prompts_####
# =============================================================================
# from prompt_synt_data_maker import PromptSyntDataMaker
# 
# dict_ = {
#     'body_intro': 'You are an expert Norwegian economist',
#     'task': 'process',
#     'language': 'nb',
#     'longitude': 'short',
#     'rep_format': 'lines',
#     'example': 'This is an example',
#     'objetive': 'This is a description'
# }
# 
# prompt_maker = PromptSyntDataMaker(prompt_lang='sp')
# example = """Descripción:
# Esta clase incluye la fabricación de todo tipo de productos de panadería, como pan, pasteles, otras harinas y pan de molde.
# Su respuesta:
# Fabricación de todo tipo de productos de panadería y pastelería.
# Fabricación de pan y bollos con harina integral.
# Producción de productos de panadería y pastelería.
# Producción de bollos, pasteles y todo tipo de productos de pastelería.
# Producción de pan integral y pan con semillas.
# Producción de pan y harina refinada.
# Molienda de trigo para la producción de harina.
# Producción de harina para piensos.
# Producción de todo tipo de harina para panadería.
# Producción de productos de panadería con y sin semillas.
# """
# print(prompt_maker.write_prompt(
#     body_intro='Eres un economista español',
#     task='generate',
#     language='sp',
#     format_='lines',
#     longitude='medium',
#     example=example,
#     description='Esto es una descripcion'
# ))
# =============================================================================
