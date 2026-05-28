import torch


class MemoryHandler:
    def __init__(self, context_size: int, prompt_memory: int):
        self._context_size = context_size
        self._prompt_memory = prompt_memory
        self._memory = []

    def add_memory(self, new_memory):
        self._memory.extend(new_memory)

    def get_memory(self):
        if self._prompt_memory:
            if self._prompt_memory == 0:
                return_memory = []
            else:
                # each prompt memory should have a user and assistant message and we want the last _prompt_memory of these
                return_memory = self._memory[-2 * self._prompt_memory :]
        
        # # Todo: implement. Typical way is probably to tokenize the memory and only keep context_size tokens.
        # #       However, currently this process is storing text in the memory, not tokens
        # elif self._context_size:
        #     raise(NotImplementedError(""))
        #     if self._context_size > 0:
        #         return_memory = []

        return return_memory

    def get_responses(self):
        return self._memory
