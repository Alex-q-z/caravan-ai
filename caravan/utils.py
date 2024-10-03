# convert a pytorch tensor to a feature string
def feature_to_string(feature):
    feature_string = ""
    for k in range(feature.shape[0]):
        feature_string += f"{feature[k].item()},"
    feature_string = feature_string[:-1]
    return feature_string


# generate an enumerated feature string as the input to LLM-based data labeling
def generate_feature_string(features):
    prompt = ""
    for flow_index in range(features.shape[0]):
        prompt += f"({flow_index+1}) "
        prompt += feature_to_string(features[flow_index])
        prompt += " "
    return prompt


# parse generated labels from LLM API response
def parse_api_response_label(response):
    labels = []
    unparsed_labels = response.split("\n")
    for unparsed_label in unparsed_labels:
        if "(" not in unparsed_label:
            continue
        labels.append(float(unparsed_label.split(" ")[1]))
    return labels