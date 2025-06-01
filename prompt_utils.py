# Portions of this code are adapted from a project originally developed by Two Sigma Open Source, LLC,
# which is licensed under the Apache License, Version 2.0.
# 
# The original license and copyright notice are preserved below as required:
#
# Copyright 2024 Two Sigma Open Source, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from difflib import get_close_matches
from abc import ABC, abstractmethod
import random
import re
from typing import Union
from openai import OpenAI
import os
import tiktoken
from jinja2 import Environment, BaseLoader


class BaseLMProvider:
    def __init__(self):
        pass

    @abstractmethod
    def templated_prompt(self, prompt_template, input_fields, params):
        """
        Executes call to GPT using prompt, input_fields, and API parameters

        :param prompt_template: jinja template
        :param input_fields: dict of k/v pairs to fill template
        :param params: ray_cmds.Params
        :return: {'texts': [...] # API Response}
        """
        pass

class OpenAILMProvider(BaseLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initializes the OpenAILMProvider with the necessary API key and model.

        :param api_key: OpenAI API key.
        :param model: GPT model to use (default is "gpt-4o-mini").
        """
        super().__init__()
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def templated_prompt(self, prompt_template: str, input_fields: dict, params: 'Params') -> dict:
        """
        Renders the prompt using Jinja2 and sends it to the OpenAI API.

        :param prompt_template: Jinja2 template string.
        :param input_fields: Dictionary of key-value pairs to fill the template.
        :param params: Instance of Params containing API call parameters.
        :return: Dictionary containing the API response texts.
        """
        # Render the prompt using Jinja2
        try:
            env = Environment(loader=BaseLoader())
            template = env.from_string(prompt_template)
            rendered_prompt = template.render(**input_fields)
        except Exception as e:
            print("====== Rendered Template Debug ======")
            print(prompt_template)
            print("=====================================")
            raise e

        try:
            # Make the API call to OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": rendered_prompt}
                ],
                max_tokens=params['provider_args']['max_tokens'],
                n=params['provider_args']['n'],
            )

            # Extract the generated texts from the response
            texts = [choice.message.content for choice in response.choices]

            return {"texts": texts}

        except Exception as e:
            # Catch-all for unexpected errors
            print(f"Unexpected Error: {e}")
            return {"texts": [], "error": "An unexpected error occurred."}


def get_lm_provider(model_name: str = "gpt-4o-mini") -> BaseLMProvider:
    """
    Returns an instance of a subclass of BaseLMProvider based on the model name.

    :param model_name: Name of the model (e.g., "gpt-4o-mini").
    :return: Instance of BaseLMProvider subclass.
    """
    if model_name.startswith("gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return OpenAILMProvider(api_key=api_key, model=model_name)
    else:
        raise ValueError(f"Model {model_name} not supported.")


def count_tokens_in_request(lm_provider: BaseLMProvider, request: str) -> int:
    """
    Counts the number of tokens in the prompt based on the model's tokenizer.

    :param lm_provider: Instance of BaseLMProvider (e.g., OpenAILMProvider).
    :param request: The rendered prompt string.
    :return: Integer count of tokens.
    """
    if isinstance(request, list):
        request = "\n".join(request)
    if isinstance(lm_provider, OpenAILMProvider):
        # Initialize the tokenizer for the specific model
        encoding = tiktoken.encoding_for_model(lm_provider.model)
        tokens = encoding.encode(request)
        return len(tokens)
    else:
        raise NotImplementedError("Token counting not implemented for this LM provider.")

def sanitize_jinja_tokens(s):
    if isinstance(s, str):
        return s.replace("{{", "{ {").replace("}}", "} }")
    return s


def executor(lm_provider, prompt, params={}):
    """
    Executes a prompt using a LM Provider

    :param lm_provider: BaseLMProvider
    :param prompt: string prompt as jinja template string
    :param params: k/v pairs for the prompt
    :return: tuple(list[str] # list of responses, str # prompt filled with params if specified)
    """
    prompt = sanitize_jinja_tokens(prompt)
    d = lm_provider.templated_prompt(
        prompt_template=prompt,
        input_fields={},
        params=params
    )
    return d['texts'], prompt


LM_MODEL = 'gpt-4o-mini'

def get_model():
    return get_lm_provider(LM_MODEL)


LM_LONG_MODEL = 'gpt-4.1-nano'

def get_long_model():
    return get_lm_provider(LM_LONG_MODEL)


def fix_code_prompt(string):
    """
    Returns a prompt to fix a non-compiling code response from the LLM

    :param string: non-compiling code response as a string
    :return: string prompt
    """
    code, error = string
    prompt_str = \
        """You have generated some python code for me, but its not working when i run it through the exec() function. I will give you the
code (enclosed in +++) and the error (enclosed in +++) and its your job to give me back the corrected code without any errors. Assume
that the classes: CategoricalSemanticType, NumericSemanticTypeWithUnits, NumericSemanticType, BooleanSemanticType, GenericSemanticType have already been defined.
- IMPORTANT: make the MINIMAL number of edits necessary to fix the code

Here is an example of what I want you to do:

CODE=+++
class serialnumber(NumericSemanticType):
    def __init__(self):
        self.description = "Serial Numbers"
        self.valid_range = [1, float('inf')]
        self.dtype = int
        self.format = "Serial numbers should be positive integers"
        self.examples = [1, 2, 3, 4, 5]
```

```python
MAPPING = {'Sno': serialnumber}
+++
ERROR=+++
invalid syntax (<string>, line 38)
+++
FIXED=+++
class serialnumber(NumericSemanticType):
    def __init__(self):
        self.description = "Serial Numbers"
        self.valid_range = [1, float('inf')]
        self.dtype = int
        self.format = "Serial numbers should be positive integers"
        self.examples = [1, 2, 3, 4, 5]
MAPPING = {'Sno': serialnumber}
+++
    
Now I want you to do the same here:
CODE=+++
{{code}}
+++
ERROR=+++
{{error}}
+++
FIXED=
"""
    definition_prompt_template = Environment(loader=BaseLoader).from_string(prompt_str)

    next_prompt = definition_prompt_template.render(
        {
            'code': code,
            'error': error,
        }
    )

    return next_prompt


def test_exec(string):
    """
    Tests execution of code string

    :param string: code string
    :return: boolean if it compiles
    """
    try:
        exec(string)
        return True
    except Exception as e:
        return str(e)


def quick_doctor(code):
    """
    Quickly applies regex fixes to the string to get it to compile
    """
    code = code.replace("n't", 'nt')
    exec_output = test_exec(code)
    if not isinstance(exec_output, str):
        return code
    else:
        ret = re.search(r"```([\s\S]*)```", code)
        if ret is None:
            return None
        extract = ret.group(1)
        exec_output = test_exec(extract)
        if not isinstance(exec_output, str):
            return extract
        return None


def fix_code(code_and_error, gpt_params, use_gpt=True):
    """
    Executes code fixing

    :param code_and_error: Tuple containing code and error message
    :param gpt_params: Instance of Params containing GPT parameters
    :param use_gpt: Boolean indicating whether to use GPT for fixing
    :return: Fixed code string or original code if fixing fails
    """
    lm_provider = get_model()
    code, _ = code_and_error
    if not isinstance(test_exec(code), str):
        return code

    quick_fix = quick_doctor(code)
    if quick_fix is not None:
        return quick_fix

    ret = re.search(r"```python(?:\n)?([\s\S]*)(?:```|\n)", code)
    if ret is None:
        if not use_gpt:
            return code_and_error
        else:
            str_prompt = fix_code_prompt(code_and_error)
            fixed_code, _ = executor(
                lm_provider,
                str_prompt,
                {
                    "provider_args": {
                        "max_tokens": gpt_params.MAX_TOKENS,
                        "n": gpt_params.BATCH_SIZE
                    }
                }
            )
            return format_code_output(fixed_code[0])
    else:
        potensh_code = ret.group(1)
        exec_output = test_exec(potensh_code)
        if not isinstance(exec_output, str):
            return potensh_code
        else:
            if not use_gpt:
                return code_and_error
            else:
                code_and_error = [potensh_code, exec_output]
                str_prompt = fix_code_prompt(code_and_error)
                fixed_code, _ = executor(
                    lm_provider,
                    str_prompt,
                    {
                        "provider_args": {
                            "max_tokens": gpt_params.MAX_TOKENS,
                            "n": gpt_params.BATCH_SIZE
                        }
                    }
                )
                return format_code_output(fixed_code[0])


def format_code_output(string):
    """
    Some basic string stripping to fix code string compilation
    """
    return string.strip('`python"+')
