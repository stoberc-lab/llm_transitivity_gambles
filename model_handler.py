from benchmark_logger import get_logger
from memory_handler import MemoryHandler
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
from transformers import pipeline
import torch
import itertools
import json
import os
from copy import deepcopy


class ModelHandler:
    def __init__(
        self,
        model_name: str,
        model_revision: str,
        base_dir: str,
        benchmark: str,
        hf_cache_dir: str,
    ):
        self._logger = get_logger(base_dir, benchmark, self.__class__.__name__)

        self._model_name = model_name
        self._model_revision = model_revision
        self._benchmark = benchmark
        self._hf_cache_dir = hf_cache_dir
        self._base_dir = base_dir

        self._device = "cpu"
        if torch.cuda.is_available():
            self._device = "cuda"

        self._get_hf_access_token()

        self._model_runs = 0

    # Read HuggingFace API access token from local file
    # This file and the token should NEVER be uploaded to git or shared with others
    # This token identifies the account of the person who created it and allows operations to be performed as that user
    # This token is needed if a model to be used requires special permissions to use, such as the META Llama 2 and 3 models
    def _get_hf_access_token(self):
        access_token = ""
        access_token_file_paths = [
            Path(f"~/huggingface/.hf_access_token"),
            Path(f"{self._base_dir}/.hf_access_token"),
        ]

        for access_token_file_path in access_token_file_paths:
            if access_token_file_path.is_file():
                with open(access_token_file_path) as access_token_file:
                    access_token = access_token_file.read()
                if access_token == "" or not access_token.startswith("hf_"):
                    self._logger.warning(
                        f"Invalid HuggingFace access token found at '{access_token_file_path}'"
                    )
                    access_token = ""

        if access_token == "":
            self._logger.warning(
                f"HuggingFace access token not found. Expected to exist at one of: {access_token_file_paths}. "
                "Attempting to continue benchmark. If the model repo requires authorization, this will fail."
            )
        self._hf_access_token = access_token.rstrip()

    def _tensorize(self, prompt, memory=[], decode=True):
        if decode:
            formatted_prompt = prompt.formatted_prompt(memory=memory)
        else:
            formatted_prompt = prompt.formatted_prompt()

        inputs = {}

        if prompt._text_format == "chat":
            continue_final_message = False
            if formatted_prompt[-1]["role"] == "assistant":
                continue_final_message = True

            formatted_chat = self._tokenizer.apply_chat_template(
                formatted_prompt,
                tokenize=True,
                continue_final_message=continue_final_message,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Fix to ensure that the tokenized message ends with a space
            # At least one chat template or code library issue is causing it to not happen
            formatted_chat_text = self._tokenizer.decode(formatted_chat[0])
            if (
                formatted_prompt[-1]["content"][-1] == " "
                and formatted_chat_text[-1] != " "
            ):
                formatted_chat = torch.cat(
                    (formatted_chat, torch.tensor([[self._model_space_char_token]])),
                    dim=-1,
                )

            inputs["input_ids"] = formatted_chat
        else:
            inputs["input_ids"] = self._tokenizer.encode(
                formatted_prompt, return_tensors="pt", return_attention_mask=True
            )

        inputs["input_ids"] = inputs["input_ids"].to(self._device)
        inputs["attention_mask"] = torch.Tensor(
            [[1.0] * len(inputs["input_ids"][0])]
        ).to(self._device)

        return inputs

    def load_model(self):
        hf_cache_dir = self._hf_cache_dir if self._hf_cache_dir else None
        if hf_cache_dir:
            self._logger.info(f"Attempting to load models from '{hf_cache_dir}'")

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            token=self._hf_access_token,
            revision=self._model_revision,
            device_map="auto",
            # local_files_only=True,
            cache_dir=hf_cache_dir,
        )

        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            revision=self._model_revision,
            token=self._hf_access_token,
            cache_dir=hf_cache_dir,
        )

        self._model_eos_token_id = self._model.config.eos_token_id
        self._model_space_char_token = self._tokenizer.encode(" ")[-1]

        if self._tokenizer.chat_template:
            self._logger.info(
                f"Found model chat template:\n"
                f"---\n{self._tokenizer.chat_template}\n---\n"
                "Input will be formatted as a chat with this template."
            )
            self.chat_model = True
        else:
            self._logger.info(
                f"No chat template found for this model. Input will be formatted as plain text."
            )
            self.chat_model = False

    def generate_text(
        self,
        prompts: list,
        max_new_tokens: int,
        seeds: list,
        temperatures: list[float],
        memory: bool = False,
        context_size: int = None,
        prompt_memory: int = None,
    ):
        if memory:
            response = self._process_with_memory_single_token(
                prompts, seeds, temperatures, context_size, prompt_memory
            )
        else:
            if len(temperatures) == 1 and temperatures[0] == -1:
                response = self._seeded_single_token_text_greedy(prompts, seeds)
            else:
                if max_new_tokens == 1:
                    response = self._seeded_single_token_text_with_temperature(
                        prompts, seeds, temperatures
                    )
                else:
                    response = self._seeded_text_with_temperature(
                        prompts, max_new_tokens, seeds, temperatures
                    )

        return response

    def generate_probs(
        self,
        prompts: list,
        max_new_tokens: int,
        temperatures: list,
        memory: bool = False,
        context_size: int = None,
        prompt_memory: int = None,
    ):
        if memory:
            raise(
                ValueError(
                    "Use run_type = 'memory_probs' to get probabilities with memory."
                )
            )
        else:
            if max_new_tokens > 1:
                raise (
                    NotImplementedError(
                        "Generating probabilities for more than a single token is not currently implemented."
                    )
                )
            else:
                response = self._single_token_probs(prompts, temperatures)

        return response
    
    def generate_memory_probs(
        self,
        prompts: list,
        max_new_tokens: int,
        temperatures: list,
        memory: bool = False,
        context_size: int = None,
        prompt_memory: int = None,
    ):
        print("memory_probs!!!")
        if max_new_tokens > 1:
            raise (
                NotImplementedError(
                    "Generating probabilities for more than a single token is not currently implemented."
                )
            )
        else:
            response = self._process_with_memory_probs_single_token(
                prompts, temperatures, context_size, prompt_memory
            )

        return response

    def _seeded_single_token_text_with_temperature(
        self, prompts: list, seeds: list, temperatures: list
    ):
        for seed in seeds:
            set_seed(seed)
            for prompt in prompts:
                for temperature in temperatures:
                    if not prompt.is_new_prompt(seed, temperature):
                        continue

                    response_text = self._single_token_text_with_temperature(
                        prompt, temperature
                    )

                    prompt.add_response(seed, temperature, {"response": response_text})

        return prompts

    def _single_token_text_with_temperature(self, prompt, temperature, memory=[]):
        tensor = self._tensorize(prompt, memory=memory)
        response = self._model.generate(
            **tensor, do_sample=True, temperature=temperature, max_new_tokens=1
        )
        response_text = self._tokenizer.decode(response[0][-1])
        return response_text

    def _seeded_text_with_temperature(
        self, prompts, max_new_tokens, seeds, temperatures, memory=[]
    ):
        for seed in seeds:
            for prompt in prompts:
                for temperature in temperatures:
                    if not prompt.is_new_prompt(seed, temperature):
                        continue

                    set_seed(seed)
                    response_text = self._text_with_temperature(
                        prompt, max_new_tokens, temperature, memory=memory
                    )

                    prompt.add_response(seed, temperature, {"response": response_text})

        return prompts

    def _text_with_temperature(self, prompt, max_new_tokens, temperature, memory=[]):
        tensor = self._tensorize(prompt, memory=memory)
        response = self._model.generate(
            **tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return self._tokenizer.decode(response[0])

    def _seeded_single_token_text_greedy(self, prompts: list, seeds: list):
        for seed in seeds:
            set_seed(seed)

            for prompt in prompts:
                if not prompt.is_new_prompt(seed):
                    continue
                response_text = self._single_token_text_greedy(prompt)
                prompt.add_response(seed, -1, {"response": response_text})

        return prompts

    def _single_token_text_greedy(self, prompt, memory=None):
        tensor = self._tensorize(prompt, memory=memory)
        response = self._model.generate(
            **tensor,
            do_sample=False,
            max_new_tokens=1,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
        )
        return self._tokenizer.decode(response[0][-1])

    def _single_token_probs(self, prompts: list, temperatures: list):
        for temperature in temperatures:
            self._logger.info(
                f"Elliciting probability distributions for {len(prompts)} prompts at temperature = {temperature}"
            )
            for prompt in prompts:
                if not prompt.is_new_prompt(-1, temperature):
                    continue
                prompt.add_probs(
                    -1, temperature, self._single_token_prob(prompt, temperature)
                )

        return prompts

    def _single_token_prob(self, prompt, temperature, min_p=0.00001, top_k=None, memory=None):
        prompt_tensor = self._tensorize(prompt, memory=memory)

        # print(f"Running full prompt: {repr(self._tokenizer.batch_decode(prompt_tensor['input_ids']))}")

        with torch.no_grad():
            response = self._model(**prompt_tensor)

        self._model_runs += 1

        last_token_logits = response.logits[0, -1, :]

        self._logger.debug(f"Logits before temp: {last_token_logits[:5]}")
        scores = last_token_logits / temperature
        self._logger.debug(f"Scores after temp ({temperature}): {scores[:5]}")

        probs = torch.softmax(scores, dim=0, dtype=torch.float64)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        continue_zip = True
        start_index = 0
        end_index = 0
        zip_size = 10
        token_probs = {}

        while continue_zip and start_index < len(sorted_probs):
            end_index = min(len(sorted_probs), start_index + zip_size)
            if top_k:
                end_index = min(top_k, end_index)

            sliced_probs = sorted_probs[start_index:end_index]
            sliced_indices = sorted_indices[start_index:end_index]

            start_index += zip_size

            for prob, token in zip(sliced_probs, sliced_indices):
                if prob.item() < min_p:
                    continue_zip = False
                    break

                token_probs[token.item()] = (
                    self._tokenizer.decode([token.item()]),
                    prob.item(),
                )

        return token_probs

    def _process_with_memory_single_token(
        self, prompts, seeds, temperatures, context_size, prompt_memory
    ):
        total_prompts = len(seeds) * len(prompts)
        completed_prompts = 0

        with torch.no_grad():
            for temperature in temperatures:
                if temperature in ["-1", -1]:
                    generate_kwargs = {
                        "do_sample": False,
                        "temperature": None,
                        "top_p": None,
                        "min_p": None,
                    }
                else:
                    generate_kwargs = {"do_sample": True, "temperature": temperature}

                for seed in seeds:
                    self._logger.info(f"Beginning seed '{seed}'")

                    prompt_sets = {}
                    for prompt in prompts:
                        prompt_key = prompt.key()
                        if prompt_key not in prompt_sets:
                            prompt_sets[prompt_key] = []
                        prompt_sets[prompt_key].append(prompt)

                    for prompt_key, prompt_set_prompts in prompt_sets.items():
                        self._logger.info(
                            f"Beginning prompt_key '{prompt_key}'. Completed {completed_prompts}/{total_prompts} prompts."
                        )
                        if seed != -1:
                            set_seed(seed)

                        memory_handler = MemoryHandler(context_size, prompt_memory)

                        for prompt_num, prompt in enumerate(prompt_set_prompts):
                            prompt.is_new_prompt(seed, temperature=temperature)

                            if temperature == -1:
                                generated_single_token_text = (
                                    self._single_token_text_greedy(
                                        prompt, memory=memory_handler.get_memory()
                                    )
                                )
                            else:
                                generated_single_token_text = (
                                    self._single_token_text_with_temperature(
                                        prompt,
                                        temperature,
                                        memory=memory_handler.get_memory(),
                                    )
                                )

                            # Convert the unstructured response text back into the structured format
                            new_memory = deepcopy(prompt.prompt)
                            if new_memory[-1]["role"] == "assistant":
                                new_memory[-1]["content"] += generated_single_token_text
                            else:
                                raise (Exception("HMMM"))
                            self._logger.debug(f"New memory: {new_memory}")

                            completed_prompts += 1

                            prompt.add_response(
                                seed,
                                temperature,
                                {
                                    "response": new_memory[-1]["content"],
                                    "full_prompt": prompt.formatted_prompt(
                                        memory=memory_handler.get_memory()
                                    ),
                                },
                            )
                            memory_handler.add_memory(new_memory)

        return prompts

    def _process_with_memory_probs_single_token(
        self, prompts, temperatures, context_size, prompt_memory
    ):
        with torch.no_grad():
            for temperature in temperatures:
                if temperature in ["-1", -1]:
                    generate_kwargs = {
                        "do_sample": False,
                        "temperature": None,
                        "top_p": None,
                        "min_p": None,
                    }
                else:
                    generate_kwargs = {"do_sample": True, "temperature": temperature}

                prompt_sets = {}
                for prompt in prompts:
                    prompt_key = prompt.key()
                    if prompt_key not in prompt_sets:
                        prompt_sets[prompt_key] = []
                    prompt_sets[prompt_key].append(prompt)

                for prompt_key, prompt_set_prompts in prompt_sets.items():
                    memory_handler = MemoryHandler(context_size, prompt_memory)

                    for prompt_num, prompt in enumerate(prompt_set_prompts):
                        if not prompt.is_new_prompt(-1, temperature):
                            continue

                        probs = self._single_token_prob(prompt, temperature, memory=memory_handler.get_memory())
                        # probs = {26964: ('Хронологи', 0.7629088596776415), 17835: ('Станов', 0.02414445101790097)}

                        # Convert the unstructured response text back into the structured format
                        new_memory = deepcopy(prompt.prompt)
                        if new_memory[-1]["role"] == "assistant":
                            new_memory[-1]["content"] += probs[list(probs.keys())[0]][0]
                        else:
                            raise(Exception("HMMM"))
                        
                        self._logger.debug(f"New memory: {new_memory}")

                        prompt.add_response(
                            -1,
                            temperature,
                            {
                                "response": new_memory[-1]["content"],
                                "full_prompt": prompt.formatted_prompt(
                                    memory=memory_handler.get_memory()
                                ),
                                "probs": probs
                            },
                        )
                        memory_handler.add_memory(new_memory)

        return prompts

    def _process_with_pipeline(self, prompts, max_new_tokens, seeds):
        with torch.no_grad():
            pipe = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                do_sample=True,
                max_new_tokens=max_new_tokens,
            )

            skip_count = 0
            run_count = 0
            for prompt in prompts:
                for seed in seeds:
                    if seed != -1:
                        set_seed(seed)

                    if not prompt.is_new_prompt(seed):
                        skip_count += 1
                        continue
                    run_count += 1

                    prompt_text = prompt.formatted_prompt()

                    prompt.add_response(
                        seed,
                        {
                            "response": pipe(prompt_text),
                            "full_prompt": prompt_text,
                        },
                    )
        self._logger.info(
            f"Ran a total of {run_count} prompts. Skipped {skip_count} prompts as output data already exists for them."
        )
        return prompts
