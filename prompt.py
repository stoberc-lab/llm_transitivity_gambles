import torch
from copy import deepcopy
import hashlib
import json
import os
import time


class Prompt:
    def __init__(
        self,
        model_name,
        model_revision,
        base_dir,
        benchmark,
        base_output_dir,
        test_group: str,
        test_name: str,
        text_format,
        prompt_name: str,
        prompt_base: list,
        option_keys: list,
        prompt_index: int,
    ):
        self._model_name = model_name
        self._model_revision = model_revision
        self._benchmark = benchmark
        self._base_output_dir = base_output_dir

        self._test_group = str(test_group)
        self._test_name = str(test_name)

        self._text_format = text_format

        self._prompt_name = str(prompt_name)
        self._prompt_base = prompt_base
        self._system_prompt = []

        self._option_keys = list(option_keys)
        self._prompt_index = prompt_index

        self.prompt = None
        self._prompt_uuids = {}
        self.tokens_list = []
        self._responses = {}
        self._token_probs = {}

        self._prepare_output_file_paths()

    def __str__(self):
        return str(f"{self._test_group}|{self._test_name}|{self._prompt_name}")

    def __repr__(self):
        return str(self.__dict__)

    def _prepare_output_file_paths(self):
        self._prepare_model_name()

        # the paths will need the file extension added
        # this is left off to allow for different formats to be passed to output()
        self._base_output_path = f"{self._base_output_dir}/{self._model}"

    def _prepare_model_name(self):
        self._model = f"{self._model_name.replace('/', '__')}__{self._model_revision}"

    def is_new_prompt(self, seed: int, temperature: float = 0):
        current_uuid = self._uuid(seed, temperature)

        self._prompt_uuids[f"{self._text_format}-{seed}"] = current_uuid

        if temperature == 0:
            existing_output_file_path = f"{self._base_output_path}/{seed}/{self._text_format}/{self._prompt_index}/{self._test_group}__{self._test_name}__{self._prompt_name}.out"
        else:
            existing_output_file_path = f"{self._base_output_path}/{seed}/{self._text_format}/{self._prompt_index}/{self._test_group}__{self._test_name}__{self._prompt_name}_{self._temperature_string(temperature)}.out"

        if not os.path.isfile(existing_output_file_path):
            return True

        with open(existing_output_file_path, "r") as existing_output_file:
            existing_uuid = existing_output_file.readline().rstrip()

        if current_uuid == existing_uuid:
            return False
        else:
            return True

    def formatted_prompt(self, memory=[], tree_text=None):
        if not memory:
            memory = []

        return_prompt = deepcopy(self._system_prompt + memory + self.prompt)

        if tree_text:
            return_prompt[-1]["content"] += tree_text

        if self._text_format == "base":
            self._formatted_prompt = "\n".join([x["content"] for x in return_prompt])
        elif self._text_format == "chat":
            self._formatted_prompt = return_prompt

        return self._formatted_prompt

    def key(self):
        return f"{self._test_group}__{self._test_name}__{self._text_format}"

    def _uuid(self, seed, temperature):
        """uuid to determine if results for a unique prompting have already been generated

        Args:
            seed (_type_): _description_
            temperature (_type_): _description_

        Returns:
            _type_: _description_
        """
        prompt_dict = {
            "benchmark": self._benchmark,
            "model_name": self._model_name,
            "model_revision": self._model_revision,
            "test_group": self._test_group,
            "test_name": self._test_name,
            "seed": seed,
            "prompt_name": self._prompt_name,
            "option_keys": self._option_keys,
            "text_format": self._text_format,
            "prompt": self.prompt,
        }

        if temperature != 0:
            prompt_dict["temperature"] = temperature

        dhash = hashlib.md5()

        # We need to sort arguments so {'a': 1, 'b': 2} is the same as {'b': 2, 'a': 1}
        encoded = json.dumps(prompt_dict, sort_keys=True).encode()
        dhash.update(encoded)

        return dhash.hexdigest()

    def _temperature_string(self, temperature):
        temperature_string = f"temperature-{temperature}"
        if temperature in ["-1", -1]:
            temperature_string = "greedy"
        return temperature_string

    def outputs(self, out_format):
        outputs_dict = {}
        for seed, temperatures_dict in self._responses.items():
            for temperature, response_dict in temperatures_dict.items():
                if out_format == "csv":
                    filename = f"{self._base_output_path}.csv"
                elif out_format == "out":
                    if temperature == 0:
                        filename = f"{self._base_output_path}/{seed}/{self._text_format}/{self._prompt_index}/{self._test_group}__{self._test_name}__{self._prompt_name}.out"
                    else:
                        filename = f"{self._base_output_path}/{seed}/{self._text_format}/{self._prompt_index}/{self._test_group}__{self._test_name}__{self._prompt_name}_{self._temperature_string(temperature)}.out"
                else:
                    raise (Exception(f"Unexpected output file format: '{out_format}'"))
                if filename not in outputs_dict:
                    outputs_dict[filename] = []

                if out_format == "csv":
                    new_output = {
                        "response_date": response_dict["response_date"],
                        "model_name": self._model_name,
                        "model_revision": self._model_revision,
                        "seed": seed,
                        "test_group": self._test_group,
                        "test_name": self._test_name,
                        "prompt_name": self._prompt_name,
                        "text_format": self._text_format,
                        "prompt_index": self._prompt_index,
                        "option_keys": self._option_keys,
                        "options": self.options,
                    }
                    if temperature != 0:
                        new_output["temperature"] = temperature
                    outputs_dict[filename].append(new_output)
                elif out_format == "out":
                    uuid_key = f"{self._text_format}-{seed}"
                    outputs_dict[filename].append(
                        "\n".join(
                            [
                                f"{self._prompt_uuids[uuid_key]}",
                                "-----",
                                f"Benchmark: {self._benchmark}",
                                f"Model: {self._model_name} - {self._model_revision}",
                                f"Seed: {seed}",
                                f"Temperature: {temperature}",
                                f"Test group: {self._test_group}",
                                f"Test name: {self._test_name}",
                                f"Prompt name: {self._prompt_name}",
                                f"Text format: {self._text_format}",
                                f"Prompt index: {self._prompt_index},"
                                f"Prompt options: {self.options}",
                                f"Prompt option keys: {self._option_keys}",
                                f"Prompt input:",
                                f"{self.prompt}",
                                "",
                            ]
                        )
                    )
                    if "full_prompt" in response_dict:
                        outputs_dict[filename][-1] += (
                            "Full prompt:\n"
                            f"{json.dumps(response_dict['full_prompt'])}\n"
                            "\n"
                        )

                if out_format == "csv":
                    if "probs" in response_dict:
                        outputs_dict[filename][-1]["probs"] = json.dumps(
                            response_dict["probs"]
                        )
                    else:
                        outputs_dict[filename][-1]["answer"] = response_dict["response"]
                elif out_format == "out":
                    outputs_dict[filename][-1] += (
                        f"Prompt response:\n" f"{response_dict}\n" "-----\n\n"
                    )

        return outputs_dict

    def _extract_last_character_answer(self, response):
        generated_text = response[0]["generated_text"]
        if type(generated_text) is list:
            return generated_text[-1]["content"][-1]
        elif type(generated_text) is str:
            return generated_text[-1]
        else:
            raise (
                Exception(
                    f"Expected response generated_text to be a list or a str. Response: '{response}'"
                )
            )

    def apply_template_options(self, options: list[dict | str]):
        prompt = deepcopy(self._prompt_base)
        self.options = options

        for option_num, option_val in enumerate(options):
            template_code = f"OPT_{option_num+1}"
            for message_num in range(len(prompt)):
                if type(option_val) is dict:
                    for key, val in option_val.items():
                        if key == "choices" and type(val) is list:
                            for choice_num, choice_val in enumerate(val):
                                template_code = f"OPT_{choice_num+1}"

                                prompt[message_num]["content"] = prompt[message_num][
                                    "content"
                                ].replace(
                                    f"{template_code}_KEY",
                                    f"{self._option_keys[choice_num]}",
                                )

                                if isinstance(choice_val, dict):
                                    for inner_key, inner_val in choice_val.items():
                                        prompt[message_num]["content"] = prompt[
                                            message_num
                                        ]["content"].replace(
                                            f"{template_code}_CHOICE.{inner_key.upper()}",
                                            f"{inner_val}",
                                        )
                                prompt[message_num]["content"] = prompt[message_num][
                                    "content"
                                ].replace(f"{template_code}_CHOICE", f"{choice_val}")
                        else:
                            prompt[message_num]["content"] = prompt[message_num][
                                "content"
                            ].replace(
                                f"{template_code}_KEY",
                                f"{self._option_keys[option_num]}",
                            )
                            prompt[message_num]["content"] = prompt[message_num][
                                "content"
                            ].replace(f"{template_code}_CHOICE.{key.upper()}", f"{val}")
                else:
                    prompt[message_num]["content"] = prompt[message_num][
                        "content"
                    ].replace(
                        f"{template_code}_KEY", f"{self._option_keys[option_num]}"
                    )
                    prompt[message_num]["content"] = prompt[message_num][
                        "content"
                    ].replace(f"{template_code}_CHOICE", f"{option_val}")

        if prompt[0]["role"] == "system":
            self._system_prompt = [prompt[0]]
            del prompt[0]

        self.prompt = prompt

    def add_response(self, seed: int, temperature: float, response: dict):
        response["response_date"] = time.time()
        if seed not in self._responses:
            self._responses[seed] = {}
        self._responses[seed][temperature] = response

    def add_probs(self, seed: int, temperature: float, token_probs: dict):
        if seed not in self._responses:
            self._responses[seed] = {}
        response = {"response_date": time.time(), "probs": token_probs}

        self._responses[seed][temperature] = response
