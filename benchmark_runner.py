from benchmark_logger import get_logger
from test_handler import TestHandler
from model_handler import ModelHandler
from output_handler import OutputHandler
from prompt import Prompt
from pathlib import Path
import itertools
import copy
import json
import os
import copy

class BenchmarkRunner:
    def __init__(self, **kwargs):
        # defaults, can be overwritten by kwargs
        self._context_size = -1
        self._prompt_memory = 0
        self._temperatures = [0]

        self.__dict__.update((f"_{k}", v) for k, v in kwargs.items())

        self._logger = get_logger(
            self._base_dir, self._benchmark, self.__class__.__name__
        )

        self._model_handler = ModelHandler(
            self._model_name,
            self._model_revision,
            self._base_dir,
            self._benchmark,
            self._hf_cache_dir,
        )

        self._test_handler = TestHandler(
            self._base_dir, self._benchmark, self._group, self._test, self._seeds
        )
        self._load_benchmark_config()

        self._base_output_dir = (
            f"{self._base_dir}/results/{self._benchmark}/{self._config_name}"
        )

        self._output_handler = OutputHandler(
            self._model_name,
            self._model_revision,
            self._base_dir,
            self._benchmark,
            self._base_output_dir,
        )

    def _load_benchmark_config(self):
        # load from config json file
        # try the config value as a path first
        config_path = self._config
        if not os.path.isfile(config_path):
            # try the config value as a file name in the base dir
            config_path = f"{self._base_dir}/{self._config}.json"
            if not os.path.isfile(config_path):
                raise (
                    Exception(
                        f"Unable to find config file at either '{self._config}' or '{config_path}."
                    )
                )
        with open(config_path) as config_file:
            benchmark_config = json.load(config_file)
            self._logger.info(json.dumps(benchmark_config))

        # values set at command line should take precedence over those in the config file
        for key in benchmark_config:
            if key in self.__dict__:
                del benchmark_config[key]

        self.__dict__.update((f"_{k}", v) for k, v in benchmark_config.items())

        expected_config_values = [
            "_memory",
            "_permutations",
            "_run_type",
            "_num_tokens",
        ]

        memory_settings_error = False
        if self._memory:
            if (
                "_context_size" not in self.__dict__
                and "_prompt_memory" not in self.__dict__
            ):
                memory_settings_error = True

        if memory_settings_error or not all(
            expected_config_value in self.__dict__
            for expected_config_value in expected_config_values
        ):
            raise (
                Exception(
                    f"Config data is expected to contain values for the following fields: {expected_config_values}.\nConfig is: '{benchmark_config}'."
                )
            )

        ### being config validation
        if self._run_type in ["probs", "probabilities"]:
            if -1 in self._temperatures:
                raise (
                    ValueError(
                        "A temperature of -1, indicating greedy search, is not valid in the context of elliciting token probability distributions. "
                        f"Run type: {self._run_type}. Temperatures: {self._temperatures}"
                    )
                )
        if 0 in self._temperatures:
            raise (
                ValueError(
                    f"A temperature of 0 is invalid in all contexts. Run type: {self._run_type}. Temperatures: {self._temperatures}"
                )
            )

        ### end config validation
        self._config_name = Path(config_path).stem

    def _is_memoried_test(self):
        return self._memory

    def run(self):
        self._model_handler.load_model()

        prompts = self._get_prompts(
            self._test_handler.tests(), self._model_handler.chat_model
        )

        responses = []
        for prompt_format_name, format_prompts in prompts.items():
            if self._run_type in ["probabilities", "probs"]:
                if self._seeds != [-1]:
                    self._logger.warn(
                        f"Probability ellicitation method is unaffected by seed. Ellicitation will not be repeated for multiple seeds. Seeds: {self._seeds}"
                    )
                self._logger.info(
                    f"Generating probabilities for: model '{self._model_name}', "
                    f"prompt format '{prompt_format_name}', {len(format_prompts)} prompts, "
                    f"{self._num_tokens} max new tokens, {len(self._temperatures)} temperatures"
                )
                responses = self._model_handler.generate_probs(
                    format_prompts,
                    self._num_tokens,
                    self._temperatures,
                    self._memory,
                    self._context_size,
                    self._prompt_memory,
                )
            elif "text" == self._run_type:
                self._logger.info(
                    f"Generating text for: model '{self._model_name}', "
                    f"prompt format '{prompt_format_name}', {len(self._test_handler.seeds())} seeds, "
                    f"{len(format_prompts)} prompts, {self._num_tokens} max new tokens, {len(self._temperatures)} temperatures"
                )
                responses = self._model_handler.generate_text(
                    format_prompts,
                    self._num_tokens,
                    self._test_handler.seeds(),
                    self._temperatures,
                    self._memory,
                    self._context_size,
                    self._prompt_memory,
                )

            # For each prompt, get logprobs, add question/response to memory using highest prob option
            elif "memory_probs" == self._run_type:
                if self._seeds != [-1]:
                    self._logger.warn(
                        f"Probability ellicitation method is unaffected by seed. Ellicitation will not be repeated for multiple seeds. Seeds: {self._seeds}"
                    )
                self._logger.info(
                    f"Generating memoryfull probabilities for: model '{self._model_name}', "
                    f"prompt format '{prompt_format_name}', {len(format_prompts)} prompts, "
                    f"{self._num_tokens} max new tokens, {len(self._temperatures)} temperatures"
                )
                responses = self._model_handler.generate_memory_probs(
                    format_prompts,
                    self._num_tokens,
                    self._temperatures,
                    self._memory,
                    self._context_size,
                    self._prompt_memory,
                )

            self._logger.info(
                f"Outputting responses from {len(responses)} prompts to results files."
            )
            self._output_handler.output_to_test_format_files(responses)

    def _get_prompts(self, tests, chat_model):
        self._logger.info("Getting prompts")
        prompts = {}
        count = 0

        text_formats = ["base"]
        if chat_model:
            text_formats.append("chat")

        for (
            group_name,
            prompt_template,
            options_file_name,
            options_data,
        ) in tests:
            option_keys = options_data["option_keys"]

            if self._permutations:
                self._logger.info(
                    f"Generating prompts with permutations of {len(options_data['options'])} test options and choice set size of {len(option_keys)}"
                )
                options_list = list(
                    itertools.permutations(options_data["options"], r=len(option_keys))
                )
            else:
                options_list = options_data["options"]

            for prompt_format in prompt_template:
                prompt_index = 1
                format_prompts = []

                prompt_name = prompt_format["prompt_name"]
                if self._prompt_name != "all" and self._prompt_name != prompt_name:
                    continue

                prompt_base = prompt_format["prompt"]

                for current_options in options_list:
                    for text_format in text_formats:
                        prompt = Prompt(
                            self._model_name,
                            self._model_revision,
                            self._base_dir,
                            self._benchmark,
                            self._base_output_dir,
                            group_name,
                            options_file_name,
                            text_format,
                            prompt_name,
                            prompt_base,
                            option_keys,
                            str(prompt_index).zfill(len(str(len(options_list)))),
                        )
                        prompt.apply_template_options(current_options)
                        format_prompts.append(prompt)
                    prompt_index += 1

                    prompts[
                        f"{text_format}_{group_name}_{options_file_name}_{prompt_name}"
                    ] = format_prompts
                count += len(format_prompts)

        self._logger.info(f"Gathered {count} total prompts.")
        return prompts

    def _model_generate(self, inputs):
        return self._model_handler.generate(inputs)
