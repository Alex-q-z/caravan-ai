import sys
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .prompts import *
from .utils import *


class Agent:

    def __init__(self, models, application, evaluation_strategy="majority_voting"):

        def initialize_generators(models, application):
            generators = []
            for model in models:
                generators.append(Generator(model, application))
            return generators

        def initialize_evaluator(evaluation_strategy):
            return Evaluator(evaluation_strategy)

        self.models = models if isinstance(models, list) else [models]
        self.application = application
        self.evaluation_strategy = evaluation_strategy

        # initialize generators and evaluator
        self.generators = initialize_generators(self.models, self.application)
        self.evaluator = initialize_evaluator(self.evaluation_strategy)
    
    def label(self, data):
        # TODO: parallize
        for generator in self.generators:
            labels = generator.label(data)
        import pdb; pdb.set_trace()
        final_labels = self.evaluator.vote(labels)
        return final_labels


class Generator:

    def __init__(self, model, application, max_model_len=2000, temperature=0.3):
        
        def initialize_prompt_manager(application):
            if application == "intrusion_detection":
                return IntrusionDetectionPrompts()
            else:
                return NotImplementedError
        
        self.model = model
        self.application = application
        
        if "gemma" in self.model:
            os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

        # set up LLM in vLLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.llm = LLM(model=self.model, 
                       tensor_parallel_size=1,
                       max_model_len=max_model_len, 
                       trust_remote_code=True, 
                       enforce_eager=True)
        self.sampling_params = SamplingParams(temperature=temperature, 
                                              max_tokens=max_model_len, 
                                              stop_token_ids=[self.tokenizer.eos_token_id])
    
        # initialize the prompt manager
        self.prompts = initialize_prompt_manager(self.application)

    def generate(self, message):
        # apply an appropriate chat template, based on tokenizer
        prompt_token_ids = self.tokenizer.apply_chat_template(message, add_generation_prompt=True)
        output = self.llm.generate(prompt_token_ids=prompt_token_ids, 
                                   sampling_params=self.sampling_params)
        return output[0].outputs[0].text

    def label(self, data):
        feature_string = generate_feature_string(data)
        message = [
            {
                "role": "user",
                "content": self.prompts.setup_prompt() + self.prompts.labeling_prompt(feature_string),
            },
        ]
        response = self.generate(message)
        parse_labels = parse_api_response_label(response)
        return parse_labels


class Evaluator:

    def __init__(self, strategy):
        self.strategy = strategy
    
    def vote(self, labels):
        if self.strategy == "majority_voting":
            # TODO: implement majority voting
            return labels
        else:
            return NotImplementedError