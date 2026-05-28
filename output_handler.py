from benchmark_logger import get_logger
from pathlib import Path
import csv
import json
import os


class OutputHandler:
    def __init__(
        self,
        model_name: str,
        model_revision: str,
        base_dir: str,
        benchmark: str,
        base_output_dir: str,
    ):
        self._logger = get_logger(base_dir, benchmark, self.__class__.__name__)

        self._base_output_dir = base_output_dir

    def _make_dir(self, dir_str: str):
        Path(dir_str).mkdir(parents=True, exist_ok=True)

    def output_to_model_summary_file(self, prompts: list):
        outputs = {}
        for prompt in prompts:
            for filename, prompt_outputs in prompt.outputs("csv").items():
                if filename not in outputs:
                    outputs[filename] = []
                outputs[filename] += prompt_outputs
        self._write_output(outputs)

    def output_to_test_format_files(self, prompts: list):
        outputs = {}
        for prompt in prompts:
            for filename, prompt_outputs in prompt.outputs("out").items():
                if filename not in outputs:
                    outputs[filename] = []
                outputs[filename] += prompt_outputs
        self._write_output(outputs)

    def _write_output(self, outputs):
        for filename, data_list in outputs.items():
            self._make_dir(os.path.dirname(os.path.abspath(filename)))

            if type(data_list[0]) is dict:
                with open(f"{filename}", "a") as output_file:
                    fieldnames = list(data_list[0].keys())
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)

                    writer.writeheader()
                    writer.writerows(data_list)
            else:
                with open(f"{filename}", "w") as output_file:
                    output_file.writelines(data_list)
