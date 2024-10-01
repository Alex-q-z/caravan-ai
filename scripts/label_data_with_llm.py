import caravan


def main():
    
    # load dataset
    cic_ids_2017 = caravan.dataset.intrusion_detection.CICIDSDataset("/workspace/caravan-ai/datasets/cic-ids2017-example.csv")

    # initialize llm-based labeling agent
    llm_agent = caravan.agent.Agent(model_name="meta-llama/Llama-3.2-1B-Instruct")
    import pdb; pdb.set_trace()

    # start data labeling

    # compute labeling accuracy


if __name__ == "__main__":
    main()