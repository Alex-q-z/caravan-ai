# convert pytorch tensors to feature strings for LLM-based data labeling
def feature_to_string(self, feature):
    feature_string = ""
    for k in range(feature.shape[0]):
        feature_string += f"{feature[k].item()},"
    feature_string = feature_string[:-1]
    return feature_string