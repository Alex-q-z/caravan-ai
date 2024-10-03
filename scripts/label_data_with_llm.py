import caravan
from torch.utils.data import DataLoader


def main():
    
    # load dataset
    my_dataset = caravan.dataset.IntrusionDetectionDataset("/workspace/caravan-ai/datasets/unsw-nb15-example.csv")
    my_dataloader = DataLoader(my_dataset, batch_size=100, shuffle=False)

    # initialize llm-based labeling agent
    llm_agent = caravan.Agent(models="meta-llama/Llama-3.2-1B", 
                              application="intrusion_detection")

    # start iterating through the dataset
    for k, (features, gt_label) in enumerate(my_dataloader):
        
        # label data
        labals = llm_agent.label(features)

        # compute labeling accuracy
        # TODO: running average labeling accuracy stats
    

if __name__ == "__main__":
    main()