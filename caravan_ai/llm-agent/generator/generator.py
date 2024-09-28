import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from generator.prompts import *


class Generator:
    
    def __init__(self, model_name, dataset_name, max_model_len=1500, temperature=0.3):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_model_len = max_model_len
        self.temperature = temperature

        if "gemma" in self.model_name:
            os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.llm = LLM(model=self.model_name, 
                       tensor_parallel_size=1, 
                       max_model_len=self.max_model_len, 
                       trust_remote_code=True, 
                       enforce_eager=True)
        self.sampling_params = SamplingParams(temperature=self.temperature, 
                                              max_tokens=1500, 
                                              stop_token_ids=[self.tokenizer.eos_token_id])
    

    def generate_code(self, message):
        # apply an appropriate chat template, based on tokenizer
        prompt_token_ids = self.tokenizer.apply_chat_template(message, add_generation_prompt=True)
        output = self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=self.sampling_params)
        return output[0].outputs[0].text


    def generate_code_batched(self, messages):
        prompt_token_ids = [self.tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]
        import pdb; pdb.set_trace()
        outputs = self.llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=self.sampling_params)
        return [output.outputs[0].text for output in outputs]


    def generate_code_humaneval(self, problem_description_prompt):
        
        humaneval_system_prompt = advanced_instruct_prompt(problem_description_prompt)
        
        message = [
            {
                "role": "user",
                "content": humaneval_system_prompt
            },
        ]

        result = self.generate_code(message)
        
        return result

    
    def generate_code_codecontests(self, problem_description_prompt):
        
        codecontests_system_prompt = advanced_instruct_prompt(problem_description_prompt)

        message = [
            {
                "role": "user",
                "content": codecontests_system_prompt
            },
        ]

        result = self.generate_code(message)
        
        return result