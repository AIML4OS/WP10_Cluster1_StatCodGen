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

from prompt_maker import PromptMaker
import os
from data_handler import read_json


class PromptSyntDataMaker(PromptMaker):
    """
    Generate fully-resolved prompts using hierarchical template definitions
    and parameter dictionaries.

    This class extends :class:`PromptMaker` by enabling recursive template
    expansion from multiple JSON template files, and by selecting
    parameter-specific values from structured dictionaries. It is designed
    for synthetic data generation workflows where prompts must be assembled
    from reusable components.

    Parameters
    ----------
    root : str or None, optional
        Root directory containing template and parameter files. If ``None``,
        the current working directory is used. Template and parameter files will be load
        from self.root/input
    name_root_template : str, optional
        Base name of the template JSON file (without suffix). Defaults to
        ``"prompt"``, which loads ``input/prompt_template.json``. All templates must have
        the next name structure ``input/<name>_template.json``
    name_param_dict : str, optional
        Path to the JSON file containing parameter dictionaries.
        Defaults to ``"input/param_dict.json"``.
    prompt_lang : str, optional
        Language key used to load templates and parameters. Defaults to ``"en"``. Will be the prompt
        language
    template_pattern : str, optional
        Placeholder pattern used inside templates for referencing other
        template blocks. Defaults to ``"[-]"``.
    gap_ph_pattern : str, optional
        Placeholder pattern used for parameter/value substitution.
        Defaults to ``"**-**"``.

    Attributes
    ----------
    param_dict : dict
        Dictionary containing parameter values for the chosen language.
    template_dict : dict
        Dictionary mapping template names to text blocks, recursively loaded.
    prompt_template : str or None
        Current assembled prompt template (updated dynamically).
    plh_replace_map_dict : dict
        Dictionary containing resolved parameter values.
    """

    def __init__(
            self,
            root=None,
            name_root_template='prompt',
            name_param_dict='input/param_dict.json',
            prompt_lang='en',
            template_pattern='[-]',
            gap_ph_pattern='**-**'
    ):
        super().__init__(
            root=root,
            prompt_template=None,
            plh_replace_map_dict={},
            gap_ph_pattern=gap_ph_pattern
        )
        self.param_dict = {}
        if name_param_dict:
            self.param_dict = read_json(
                os.path.join(self.root, name_param_dict),
            )[prompt_lang]
        self.template_dict = self.get_dict_template(
            name=name_root_template,
            lang=prompt_lang,
            dict_template={},
            pattern=template_pattern
        )
        self.name_root_template = name_root_template
        self.template_pattern = template_pattern

    def join_templates(self, root_template: str) -> str:
        """
        Recursively fill template references inside a template string.

        Template references follow a pattern such as ``"[-]"``, where the
        hyphen represents the name of a referenced template block. For example,
        ``"[-subtemplate]"`` will be replaced by the text of ``subtemplate``.

        Parameters
        ----------
        root_template : str
            Template string possibly containing template references.

        Returns
        -------
        str
            Fully expanded template with all referenced blocks resolved.

        Raises
        ------
        ValueError
            If the pattern does not contain a hyphen, indicating an invalid
            template reference format.
        """
        PromptMaker.check_text(self.template_pattern, '-')
        gap_keys = PromptMaker.get_gap_keys(root_template, self.template_pattern)
        prompt_template = root_template
        if len(gap_keys) != 0:
            for key in gap_keys:
                prompt_template = self.fill_text(root_template, self.template_dict, self.template_pattern)
            prompt_template = self.join_templates(prompt_template)
        return prompt_template

    def get_dict_template(
            self,
            name,
            lang,
            dict_template,
            pattern
    ) -> dict:
        """
        Recursively load templates from JSON and resolve dependencies.

        For a given template name, this method loads the corresponding text
        from ``input/<name>_template.json`` and scans for referenced templates.
        It then recursively loads all dependent templates.

        Parameters
        ----------
        name : str, optional
            Template name to load. Defaults to ``"prompt"``.
        lang : str, optional
            Language key for retrieving the template text. Defaults to ``"sp"``.
        dict_template : dict, optional
            Dictionary that accumulates template name–text mappings.
        pattern : str, optional
            Pattern used for referencing other templates.

        Returns
        -------
        dict
            Dictionary mapping template names to template strings.

        Raises
        ------
        FileNotFoundError
            If the JSON template file does not exist.
        KeyError
            If the language key is not present in the template file.
        """
        json_name = f"{name}_template.json"
        json_dict = read_json(
            os.path.join(self.root, 'input', json_name)
        )
        text_template = json_dict[lang]
        dict_template[name] = text_template
        gap_keys = PromptMaker.get_gap_keys(text_template, pattern=pattern)
        if len(gap_keys) != 0:
            for key in gap_keys:
                self.get_dict_template(name=key, lang=lang, dict_template=dict_template, pattern=pattern)
        return dict_template

    def search_keys_dict(self) -> list:
        """
        Identify which placeholder keys in the template correspond to parameter entries.

        Returns
        -------
        list of str
            Keys that exist both in the template and in ``param_dict``.
        """
        key_list = self.get_gap_keys(self.prompt_template, self.gap_ph_pattern)
        key_param = self.param_dict.keys()
        return [key for key in key_list if key in key_param]

    def check_active_params_dict(self, param_keys, active_params_dict):
        for p_key in param_keys:
            v_key = active_params_dict.get(p_key)
            if v_key is not None:
                if self.param_dict[p_key].get(v_key) is None:
                    raise ValueError(f'Error getting param value "{p_key}": It must be one of the next possible values {list(self.param_dict[p_key].keys())}')

    def get_plh_replace_map_dict(self, active_params_dict: dict):
        """
        Build the dictionary of replacement values for parameter substitution.

        Parameters
        ----------
        active_params_dict : dict
            Input dictionary whose keys select values from the parameter
            dictionary.

        Notes
        -----
        - Keys present in ``param_dict`` are interpreted as indexed parameter
          selectors.
        - Keys not present in ``param_dict`` are included as-is.

        Raises
        ------
        KeyError
            If a required index key is missing in ``active_params_dict``.
        """
        param_keys = self.search_keys_dict()
        self.check_active_params_dict(param_keys, active_params_dict)
        plh_replace_map_dict = {
            key: self.param_dict[key][active_params_dict[key]]
            for key in param_keys
        }
        for key, item in active_params_dict.items():
            if key not in param_keys:
                plh_replace_map_dict[key] = item
        self.plh_replace_map_dict = plh_replace_map_dict

    def edit_template(self, active_params_dict):
        """
        Remove template lines referencing unused parameters.

        If a placeholder key exists in the template but is not provided in
        ``active_params_dict``, any line containing that placeholder is removed.

        Parameters
        ----------
        active_params_dict : dict
            Input dictionary specifying which parameters are active.
        """
        param_keys = self.get_gap_keys(
            self.prompt_template,
            self.gap_ph_pattern
        )
        sentences = self.prompt_template.split('\n')
        for param_key in param_keys:
            if active_params_dict.get(param_key) is None:
                pat = self.fill_gap(self.gap_ph_pattern, '-', param_key)
                sentences = [
                    s for s in sentences
                    if pat not in s
                ]
        self.prompt_template = "\n".join(sentences)
        self.prompt_template = self.prompt_template.replace(':', ':\n')

    def get_prompt(self, active_params_dict: dict) -> str:
        """
        Construct the final prompt with all template logic and parameter
        substitutions applied.

        Parameters
        ----------
        active_params_dict : dict
            Input dictionary selecting parameter variants and supplying
            free-form values.

        Returns
        -------
        str
            Fully rendered prompt.

        Notes
        -----
        Workflow:
        1. Expand template hierarchy.
        2. Remove unused template segments.
        3. Resolve parameter values.
        4. Fill placeholders and return final text.
        """
        self.prompt_template = self.join_templates(self.template_dict[self.name_root_template])
        self.edit_template(active_params_dict)
        self.get_plh_replace_map_dict(active_params_dict)
        return self.fill_prompt()

    def write_prompt(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs :
        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        active_params_dict = dict(**kwargs)
        return self.get_prompt(active_params_dict)
