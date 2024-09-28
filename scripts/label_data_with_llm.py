import caravan_ai


def main():
    
    # load dataset
    cic_ids_2017 = caravan_ai.dataset.intrusion_detection.CICIDSDataset("/workspace/caravan-ai/datasets/cic-ids2017-example.csv")

    # initialize llm-based labeling agent

    # start data labeling

    # compute labeling accuracy


if __name__ == "__main__":
    main()