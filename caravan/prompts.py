class Prompts:

    def __init__(self):
        self.feature_names = None
    
    def setup_prompt(self):
        return "setup prompt"
    
    def few_shot_prompt(self):
        return "few shot prompt"

    def labeling_prompt(self):
        return "labeling prompt"


class IntrusionDetectionPrompts(Prompts):

    def __init__(self):
        self.feature_names = "dur(Record total duration),proto(Transaction protocol, which will be categorized),sbytes(Source to destination transaction bytes),dbytes(Destination to source transaction bytes),sttl(Source to destination time to live value),dttl(Destination to source time to live value),sload(Source bits per second),dload(Destination bits per second),spkts(Source to destination packet count),dpkts(Destination to source packet count),smean(Mean of the packet size transmitted by the src),dmean(Mean of the packet size transmitted by the dst),sinpkt(Source interpacket arrival time (mSec)),dinpkt(Destination interpacket arrival time (mSec)),tcprtt(TCP connection setup round-trip time),synack(TCP connection setup time, the time between the SYN and the SYN_ACK packets),ackdat(TCP connection setup time, the time between the SYN_ACK and the ACK packets),ct_src_ltm(No. of connections of the same source address in 100 connections according to the last time),ct_dst_ltm(No. of connections of the same destination address in 100 connections according to the last time),ct_dst_src_ltm(No of connections of the same source and the destination address in 100 connections according to the last time)"

    def setup_prompt(self):
        return f"You are an expert in network security. I am now labeling a network intrusion detection dataset, and I want to assign a binary label (benign or malicious) to each traffic flow in the dataset based on each flow's input features. Feel free to use your own expertise and the information I give you. These are the features of the input flows and meanings of the features: {self.feature_names}. "
    
    def few_shot_prompt(self):
        return NotImplementedError

    def labeling_prompt(self, data):
        return f"Please give me a label for each of these unlabeled flows. No explanation or analysis needed, label only; One flow on each line. Format for each line: (flow number) label. {data}"


def standard_coding_prompt(problem_description: str) -> str:
    return f"""You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature).\n{problem_description}."""

def instruct_coding_prompt(problem_description: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code without any tests or explanation.\n{problem_description}\n\n### Response:"""

def detailed_coding_prompt(problem_description: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{problem_description}\n\n### Requirements:\n1. Complete the code without any tests or explanation.\n2. Include all necessary data structures and imports within the response.\n3. Return executable code, wrapped in backtick."""