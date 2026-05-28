import csv
import json
import ast
import string
import itertools
from test_handler import TestHandler
from pathlib import Path
import os
import sys
import string
from pathlib import Path

choice_table = {
    "stober_gamble_set_1": {
        "option_keys": ["1", "2"],
        "options": [
            {"money": "25.43", "fraction": "7/24", "percent": "29.17"},
            {"money": "24.16", "fraction": "8/24", "percent": "33.33"},
            {"money": "22.89", "fraction": "9/24", "percent": "37.5"},
            {"money": "21.62", "fraction": "10/24", "percent": "41.67"},
            {"money": "20.35", "fraction": "11/24", "percent": "45.83"},
        ],
    },
    "stober_gamble_set_2": {
        "option_keys": ["1", "2"],
        "options": [
            {"money": "31.99", "fraction": "7/24", "percent": "29.17"},
            {"money": "27.03", "fraction": "8/24", "percent": "33.33"},
            {"money": "22.89", "fraction": "9/24", "percent": "37.5"},
            {"money": "19.32", "fraction": "10/24", "percent": "41.67"},
            {"money": "16.19", "fraction": "11/24", "percent": "45.83"},
        ],
    },
    "tversky_gamble_set_1": {
        "option_keys": ["1", "2"],
        "options": [
            {"money": "5.00", "fraction": "7/24", "percent": "29.17"},
            {"money": "4.75", "fraction": "8/24", "percent": "33.33"},
            {"money": "4.50", "fraction": "9/24", "percent": "37.5"},
            {"money": "4.25", "fraction": "10/24", "percent": "41.67"},
            {"money": "4.00", "fraction": "11/24", "percent": "45.83"},
        ],
    },
    "tversky_gamble_set_2": {
        "option_keys": ["1", "2"],
        "options": [
            {"money": "5.0", "fraction": "8/24", "percent": "33.33"},
            {"money": "4.75", "fraction": "10/24", "percent": "41.67"},
            {"money": "4.50", "fraction": "12/24", "percent": "50.00"},
            {"money": "4.25", "fraction": "14/24", "percent": "58.33"},
            {"money": "4.00", "fraction": "16/24", "percent": "66.67"},
        ],
    },
    "tversky_gamble_set_3": {
        "option_keys": ["1", "2"],
        "options": [
            {"money": "3.70", "fraction": "7/24", "percent": "29.17"},
            {"money": "3.60", "fraction": "8/24", "percent": "33.33"},
            {"money": "3.50", "fraction": "9/24", "percent": "37.5"},
            {"money": "3.40", "fraction": "10/24", "percent": "41.67"},
            {"money": "3.30", "fraction": "11/24", "percent": "45.83"},
        ],
    },
}


def lookup_choices(test_name, choices):
    for test_set in choice_table.keys():
        if test_set in test_name:
            benchmark_data = choice_table[test_set]

    selected_option_num = ""

    choice_indices = []
    for choice in choices:
        choice_indices.append(benchmark_data["options"].index(choice))

    choice_codes = [string.ascii_uppercase[x] for x in choice_indices]

    return choice_codes


results_dir = sys.argv[1]
benchmark = sys.argv[2]
config_name = sys.argv[3]

base_dir = f"{results_dir}/{benchmark}/{config_name}"

models = [
    f.name
    for f in os.scandir(f"{base_dir}")
    if f.is_dir() and not f.name.startswith(".")
]
print(models)

data_by_prompt_name = {}

for model in models:
    model_details = model.split("__")
    print(f"Model: {model}")
    model_dir = f"{base_dir}/{model}"
    seeds = [
        f.name
        for f in os.scandir(f"{model_dir}")
        if f.is_dir() and not f.name.startswith(".")
    ]
    print(f"Seeds in model: {seeds}")
    for seed in seeds:
        seed_dir = f"{model_dir}/{seed}"
        model_input_formats = [
            f.name
            for f in os.scandir(f"{seed_dir}")
            if f.is_dir() and not f.name.startswith(".")
        ]
        for model_input_format in model_input_formats:
            model_input_format_dir = f"{seed_dir}/{model_input_format}"
            prompt_nums = [
                f.name
                for f in os.scandir(f"{model_input_format_dir}")
                if f.is_dir() and not f.name.startswith(".")
            ]
            for prompt_num in prompt_nums:
                prompt_num_dir = f"{model_input_format_dir}/{prompt_num}"
                out_files = [
                    f.name
                    for f in os.scandir(f"{prompt_num_dir}")
                    if f.is_file() and f.name.endswith(".out")
                ]
                is_response_line = False
                for out_file in out_files:
                    new_data = {
                        "seed": seed,
                        "model_name": model_details[1],
                        "model_revision": model_details[2],
                    }
                    with open(f"{prompt_num_dir}/{out_file}", "r") as cur_file:
                        file_lines = cur_file.readlines()
                        for line in file_lines:
                            line = line.strip()
                            if is_response_line:
                                response_json = ast.literal_eval(line)
                                new_data["response"] = response_json["response"]
                                is_response_line = False
                            elif line.startswith("Prompt index:"):
                                split_line = line.split(",Prompt options: ")
                                new_data["question_num"] = split_line[0].replace(
                                    "Prompt index: ", ""
                                )
                                new_data["choices"] = ast.literal_eval(split_line[1])
                            elif line.startswith("Prompt response:"):
                                is_response_line = True
                            elif line.startswith("Test group: "):
                                new_data["test_group"] = line.replace(
                                    "Test group: ", ""
                                )
                            elif line.startswith("Test name: "):
                                new_data["test_name"] = line.replace("Test name: ", "")
                            elif line.startswith("Prompt name: "):
                                new_data["prompt_name"] = line.replace(
                                    "Prompt name: ", ""
                                )
                            elif line.startswith("Text format: "):
                                new_data["text_format"] = line.replace(
                                    "Text format: ", ""
                                )
                            elif line.startswith("Temperature"):
                                new_data["temperature"] = line.replace(
                                    "Temperature: ", ""
                                )
                            else:
                                pass

                    new_data["choice_codes"] = lookup_choices(
                        new_data["test_name"], new_data["choices"]
                    )

                    if new_data["prompt_name"] not in data_by_prompt_name:
                        data_by_prompt_name[new_data["prompt_name"]] = []
                    data_by_prompt_name[new_data["prompt_name"]].append(new_data)


for prompt_name, prompt_name_data in data_by_prompt_name.items():
    print(prompt_name_data[0])
    out_path = f"./analysis/{benchmark}/{config_name}/{prompt_name}.csv"
    print(f"Writing gathered model data to: {out_path}")

    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as output_file:
        keys = [
            "model_name",
            "model_revision",
            "test_group",
            "test_name",
            "prompt_name",
            "text_format",
            "order",
            "temperature",
            "question_num",
            "seed",
            "choices",
            "choice_codes",
            "response",
        ]
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(prompt_name_data)
