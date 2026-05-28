from benchmark_logger import get_logger
from pathlib import Path
import os
import json


class TestHandler:
    def __init__(
        self,
        base_dir: str,
        benchmark: str,
        group: str,
        options_set: str,
        specific_seeds: int,
        disable_logger: bool = False,
    ):
        if not disable_logger:
            self._logger = get_logger(base_dir, benchmark, self.__class__.__name__)
        self._disable_logger = disable_logger

        self._base_dir = base_dir
        self._benchmark_dir = Path(f"{base_dir}/inputs/{benchmark}")
        self._benchmark = benchmark
        self._group = group
        self._options_set = options_set
        self._get_seeds(specific_seeds)

        self._read_test_data()

    def _read_test_data(self):
        inputs_dir = self._benchmark_dir
        benchmark_dirs = [
            f.name
            for f in os.scandir(inputs_dir)
            if f.is_dir() and not f.name.startswith(".")
        ]

        valid_file_extensions = ".json"

        test_data = {}

        # attempt to load any benchmark-wide prompt template
        benchmark_prompt_template_path = Path(f"{self._benchmark_dir}/prompt.tmpl")
        if benchmark_prompt_template_path.is_file():
            with open(benchmark_prompt_template_path) as f:
                if not self._disable_logger:
                    self._logger.info(f"Benchmark prompt template found.")
                test_data["_prompt_template"] = json.load(f)

        for group_name in benchmark_dirs:
            if self._group == "all" or (
                self._group != "all" and self._group == group_name
            ):
                test_data[group_name] = {}

                # load the prompt template
                # this file contain the general format of the prompts
                # into which the options will be substituted
                group_prompt_template_path = Path(
                    f"{self._benchmark_dir}/{group_name}/prompt.tmpl"
                )
                if group_prompt_template_path.is_file():
                    with open(group_prompt_template_path) as f:
                        if not self._disable_logger:
                            self._logger.info(
                                f"Group '{group_name}' prompt template found."
                            )
                        test_data[group_name]["_prompt_template"] = json.load(f)

                if (
                    "_prompt_template" not in test_data
                    and "_prompt_template" not in test_data[group_name]
                ):
                    raise (
                        Exception(
                            f"A prompt template must exist in either the benchmark directory '{self._benchmark_dir}' or in each group. "
                            f"No prompt template found in '{group_prompt_template_path}."
                        )
                    )

                for options_file_path in os.scandir(f"{inputs_dir}/{group_name}"):
                    options_file_name = Path(options_file_path.name).stem
                    options_file_extension = Path(options_file_path.name).suffix

                    if (
                        options_file_name == "options_template_values"
                        or options_file_name == "prompt_template"
                    ):
                        continue

                    if options_file_path.is_file() and options_file_extension in (
                        valid_file_extensions
                    ):
                        if self._options_set == "all" or (
                            self._options_set != "all"
                            and self._options_set == options_file_name
                        ):
                            with open(
                                f"{self._benchmark_dir}/{group_name}/{options_file_path.name}"
                            ) as f:
                                test_data[group_name][options_file_name] = json.load(f)

        self._test_data = test_data

    def _get_seeds(self, specific_seeds):
        # by setting a specific transformers and configuring the model correctly we can ensure deterministic generations
        # these seeds should be saved in the 'seeds.json' file in the inputs/{benchmark} directory of this project
        # or in the {base_dir}/inputs directory
        # or configured by the --seed(s) command line arg
        # seeds in the file should be parsable as [int]
        seeds = []

        seeds_default_paths = [
            f"{self._benchmark_dir}/seeds.json",
            f"{self._base_dir}/inputs/seeds.json",
        ]
        seeds_file_path = None

        if (
            specific_seeds == "all"
            or specific_seeds == "-1"
            or (
                type(specific_seeds) is list
                and len(specific_seeds) == 1
                and specific_seeds[0] in ["all", "-1"]
            )
        ):
            for seeds_default_path in seeds_default_paths:
                if Path(seeds_default_path).is_file():
                    seeds_file_path = seeds_default_path
            if not seeds_file_path:
                raise (
                    ValueError(
                        f"No seed value specified and unable to find an accessible seed file at default paths. Default paths: {seeds_default_paths}"
                    )
                )
        elif type(specific_seeds) == list:
            try:
                for item in specific_seeds:
                    seeds.append(int(item))
            except ValueError as e:
                if Path(specific_seeds[0]).is_file():
                    seeds_file_path = specific_seeds[0]
                else:
                    raise (e)
        elif Path(specific_seeds).is_file():
            seeds_file_path = specific_seeds
        else:
            try:
                seeds = [int(specific_seeds)]
            except ValueError as e:
                raise (e)

        if seeds_file_path:
            if not self._disable_logger:
                self._logger.info(f"Loading seeds from '{seeds_file_path}'.")
            with open(seeds_file_path, "r") as seeds_file:
                seeds = json.load(seeds_file)
                if type(seeds) is not list or type(seeds[0]) is not int:
                    raise (
                        Exception(
                            f"Expected seeds file ('{seeds_file_path}') to provide a list of ints. Received: '{seeds}'."
                        )
                    )

        if not seeds:
            raise (
                ValueError(
                    f"Configured seeds value is neither an accessible file nor parsable as an a List of Ints or an Int. Value: '{specific_seeds}'"
                )
            )

        self._seeds = seeds

    def seeds(self):
        return self._seeds

    def tests(self):
        """Generator for tests to be run

        Yields:
            group_name -> str: Name of the test group. From dirs under {input_dir}/{benchmark_name}/.
            prompt_template -> [dict]:
            options_file_name -> str: Name of the file the options_data is loaded from.
            options_data -> dict:
        """
        benchmark_prompt_template = (
            self._test_data["_prompt_template"]
            if "_prompt_template" in self._test_data
            else None
        )
        for group_name, group_data in self._test_data.items():
            if group_name == "_prompt_template":
                continue

            prompt_template = benchmark_prompt_template
            if "_prompt_template" in group_data:
                prompt_template = group_data["_prompt_template"]

            for options_file_name, options_data in group_data.items():
                if options_file_name == "_prompt_template":
                    continue

                yield (
                    group_name,
                    prompt_template,
                    options_file_name,
                    options_data,
                )
