# load training_data.json and make a file with 15% of that data

import json
import random

training_data = json.load(open("training_data.json"))
print("finished loading training_data")

# make a subset of the training data
new_training_data = training_data[:int(len(training_data) * 0.15)]

with open("training_data_subset.json", "w") as f:
  json.dump(new_training_data, f)
