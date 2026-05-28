import json
import itertools
from test_handler import TestHandler
import random

test_handler = TestHandler(".", "gambles", "all", "all", "all")

# set a specific seed to give recreatable orderings, set to -1 to disable and use python's default seed (system time)
RANDOMIZATION_SEED = 42561109
NUMBER_OF_DESIRED_ORDERS = 10
OUTPUT_BASE_PATH = "./inputs/gambles_memoryfull"

for (
    group_name,
    prompt_template,
    group_template_values,
    options_file_name,
    options_data,
) in test_handler.tests():
    option_keys = options_data["option_keys"]
    print(option_keys)
    options_list = list(
        itertools.permutations(options_data["options"], r=len(option_keys))
    )

    if RANDOMIZATION_SEED != -1:
        random.seed(RANDOMIZATION_SEED)

    for i in range(1, NUMBER_OF_DESIRED_ORDERS + 1):
        print(f"Generating ordering #{i}")
        if i != 0:
            print("  Shuffling choice options")
            random.shuffle(options_list)
        output = {"option_keys": option_keys, "options": []}

        for current_options in options_list:
            option_set = [{"choices": []}]
            for index in range(len(option_keys)):
                option_set[0]["choices"].append(current_options[index])
            output["options"].append(option_set)
        output_file_name = (
            f"{OUTPUT_BASE_PATH}/{group_name}/{options_file_name}_order-{i}.json"
        )
        print(f"Writing choice list to: '{output_file_name}'")
        with open(output_file_name, "w") as output_file:
            json.dump(output, output_file, indent=4)
