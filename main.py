import argparse
import sys

from benchmark_runner import BenchmarkRunner


def args_to_config(args):
    """
    Convert arguments from command line in argparse. Sets defaults and overrides with arg values.

    Args:
        args (_type_): The argparse arguments

    Returns:
        _type_: _description_
    """
    print(args)
    config = {
        "model_name": args.modelname,
        "model_revision": args.modelrevision,
        "benchmark": args.benchmark,
        "config": args.config,
        # defaults to be replaced if passed
        "group": "all",
        "test": "all",
        "seeds": "all",
        "prompt_name": "all",
        "batch_size": 16,
        "base_dir": ".",
        "hf_cache_dir": None,
    }

    if args.group:
        config["group"] = args.group
    if args.test:
        config["test"] = args.test
    if args.promptname:
        config["prompt_name"] = args.promptname
    if args.seed:
        config["seeds"] = [args.seed]
    if args.seeds:
        config["seeds"] = str(args.seeds).split(",")
    if args.batchsize:
        config["batch_size"] = args.batchsize
    if args.basedir:
        config["base_dir"] = args.basedir
    if args.hfcachedir:
        config["hf_cache_dir"] = args.hfcachedir
    if args.contextsize:
        config["context_size"] = args.contextsize
    if args.promptmemory:
        config["prompt_memory"] = args.promptmemory

    print(config)

    # to-do: verify arguments

    # if --config, read settings from yaml file

    return config


def main():
    """The initial function of the framework. Intially processing of arguments via argparse followed by starting the BenchmarkRunner."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modelname",
        required=True,
        help="Group/Name of the LLM to run. Example: 'meta-llama/Meta-Llama-3-8B'",
    )
    parser.add_argument(
        "--modelrevision",
        required=True,
        help="The commit/revision id of the LLM to execute. Example: '8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920'",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Name of the benchmark dataset to run. Data will be loaded from [basedir]/inputs/[benchmark]. Example: 'morals_memoryfull'",
    )

    config_settings = parser.add_argument_group("Config Settings")
    config_settings.add_argument(
        "--config",
        help=(
            "Path or name of json formatted test config file to read. "
            "This will define how the benchmark dataset will be used and the format of the outputs. Example: './tinyllama_config.json'"
        ),
    )
    config_settings.add_argument("--permutations", help=". Example: ''")
    config_settings.add_argument("--runtype", help=". Example: ''")
    config_settings.add_argument("--numtokens", help=". Example: ''")
    config_settings.add_argument(
        "--memory",
        help=(
            "Whether prompting is performed in sequence by including some amount of previous prompts in new prompts as 'memory' or 'context'. "
            "Boolean (true/false). Optional if supplied by --config file. Example: 'true'"
        ),
    )

    memory_settings = parser.add_argument_group(
        "Memory Settings; Only 1 should be set if --memory=true"
    )
    memory_settings.add_argument(
        "--contextsize",
        help=(
            "Number of tokens from previous history to include in new prompts. Int > 0. "
            "Required if --memory=true, Optional if supplied by --config file. Example: 30000"
        ),
    )
    memory_settings.add_argument(
        "--promptmemory",
        help=(
            "Number of previous prompts and responses to include in new prompts. Int > 0. "
            "Required if --memory=true, Optional if supplied by --config file. Example: 10"
        ),
    )

    benchmark_args = parser.add_argument_group("benchmark")
    benchmark_args.add_argument(
        "-g",
        "--group",
        help="Only run a specific test group found in the benchmark configuration. Example: 'qualtrics_order'",
    )
    benchmark_args.add_argument(
        "-t",
        "--test",
        help=(
            "Only run a specific test name found in the benchmark configuration. "
            "Can be specified with or without .json file extension. Example: 'questions'"
        ),
    )
    benchmark_args.add_argument(
        "-p",
        "--promptname",
        help="Only run a prompt format name found in the test input file(s). Example: 'fraction_with_dollars'",
    )
    benchmark_args.add_argument(
        "-s",
        "--seed",
        help=(
            "Int or Path. If a Path, load seeds from [seed] file instead of default location. "
            "If Int, only run a specific seed instead of all seeds found in the default path of "
            "'[basedir]/inputs/benchmark]/seeds.json'. Example: '42'"
        ),
    )
    benchmark_args.add_argument(
        "-S",
        "--seeds",
        help=(
            "Int or Path. If a Path, load seeds from [seed] file instead of default location. "
            "If Int, only run a specific seed instead of all seeds found in the default path of "
            "'[basedir]/inputs/benchmark]/seeds.json'. Example: '42,43'"
        ),
    )
    benchmark_args.add_argument(
        "-b",
        "--batchsize",
        help="Size of batches to send to GPU(s) for generation. This can have big impacts on performance. Default: '16'.",
        nargs="?",
        default=16,
    )
    benchmark_args.add_argument(
        "--hfcachedir",
        help=(
            "Path to directory to use for loading of downloaded files from HuggingFace, chiefly models. "
            "Defaults to the HuggingFace default."
        ),
    )
    benchmark_args.add_argument(
        "--basedir",
        help="Path to directory to use as the root for loading and saving benchmark data. Defaults to '.'.",
        default=".",
    )
    
    # parser.add_argument("-", "--", help=". Example: ''")

    config = args_to_config(parser.parse_args())

    benchmark = BenchmarkRunner(**config)

    benchmark.run()


if __name__ == "__main__":
    print(sys.argv)
    main()
