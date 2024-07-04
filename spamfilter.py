# load spam.csv
import pandas as pd
import string
import time
import torch.nn as nn
import torch
import random
import sys
import json

# load the data from training_data.json and testing_data.json

word_to_index = json.load(open("word_to_index.json"))
print("finished loading word_to_index")

index_to_word = json.load(open("index_to_word.json"))
print("finished loading index_to_word")

training_data = json.load(open("training_data_subset.json"))
print("finished loading training_data")

# convert the data to tensors

for i in range(len(training_data)):
  sentence, label = training_data[i]
  sentence = torch.tensor(sentence)
  label = torch.tensor([label])
  training_data[i] = [sentence, label]

testing_data = json.load(open("testing_data.json"))
print("finished loading testing_data")

for i in range(len(testing_data)):
  sentence, label = testing_data[i]
  sentence = torch.tensor(sentence)
  label = torch.tensor([label])
  testing_data[i] = [sentence, label]

words = json.load(open("words.json"))
print("finished loading words")

class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    self.fc1 = nn.Linear(len(words), 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.sigmoid(x)
    return x

num_epochs = 100

def train_model():
  print("training model")
  # keep track of the loss for each epoch and print it
  
  # load the model
  model = torch.load("model.pth")

  criterion = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

  # compute 10%, .... 90% of the dataset indices
  ten = len(training_data) // 10
  printing_points = [i * ten for i in range(1, 10)]


  for epoch in range(num_epochs):
    total_loss = 0
    for i in range(len(training_data)):
      sentence, label = training_data[i]
      optimizer.zero_grad()
      output = model(sentence)
      loss = criterion(output, label)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      # every 10% through the dataset, print a message
      if i in printing_points:
        print(f"{i / len(training_data) * 100}% done with epoch {epoch + 1}")
      
    print("Epoch: ", epoch + 1)
    print("Loss: ", total_loss)
    torch.save(model, "model.pth")
  
  # save the model


def make_model():
  print("making model")
  model = NN()
  torch.save(model, "model.pth")

def validate_model(model):
  correct = 0
  total = 0
  for sentence, label in testing_data:
    output = model(sentence)
    output = 1 if output.item() > 0.5 else 0
    label = label.item()
    if output == label:
      correct += 1
    total += 1

  print("Accuracy: ", correct / total)
  print("Correct: ", correct)
  print("Total: ", total)

def test_model():
  model = torch.load("model.pth")
  while True:
    sentence = input("Enter a sentence: ")
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    sentence = sentence.split()
    sentence_tensor = torch.zeros(len(words))
    for word in sentence:
      if word in word_to_index:
        sentence_tensor[word_to_index[word]] = 1

    prediction = model(sentence_tensor)
    prediction_string = "spam" if prediction.item() > 0.5 else "not spam"
    print(f"The sentence is {prediction_string}")

test_model()