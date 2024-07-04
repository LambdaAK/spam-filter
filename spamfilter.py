# load spam.csv
import pandas as pd
import string
import time
import torch.nn as nn
import torch
import random
# the first column is the label
# the second column is the text


# read from enron_spam_data.csv

csv_data = pd.read_csv("enron_spam_data.csv")


data = []

for index, row in csv_data.iterrows():
  message = row['Message']
  if str(message) == 'nan':
    continue
  label = row['Spam/Ham']
  if label == 'spam':
    label = 1
  else:
    label = 0
  data.append([str(message), label])

# turn all of the words into lowercase and get rid of punctuation

def clean_data(data):
  cleaned_data = []
  for message, label in data:
    message = message.lower()
    message = message.translate(str.maketrans('', '', string.punctuation))
    cleaned_data.append([message, label])
  return cleaned_data

cleaned_data = clean_data(data)

# split the messages into lists of words

for i in range(len(cleaned_data)):
  cleaned_data[i][0] = cleaned_data[i][0].split()

# create a list of all of the words in the dataset
  
words = set()
for message, label in cleaned_data:
  for word in message:
    words.add(word)

words = list(words)
words.sort()

# make a frequency dictionary for each word in the dataset

word_to_freq = {}
for word in words:
  word_to_freq[word] = 0

for message, label in cleaned_data:
  for word in message:
    word_to_freq[word] += 1

# print the words in order of frequency
    
word_freq = []
for word in words:
  word_freq.append([word, word_to_freq[word]])

word_freq.sort(key=lambda x: x[1], reverse=True)

# get rid of words that appear less than 10 times

for i in range(len(word_freq) - 1, -1, -1):
  if word_freq[i][1] < 20:
    word_freq.pop(i)

words = []

for word, freq in word_freq:
  words.append(word)

# create a dictionary that maps each word to an index

word_to_index = {}

for i in range(len(words)):
  word_to_index[words[i]] = i

# turn the messages into bag of words vectors
  
def message_to_vector(message):
  vector = torch.zeros(len(words))
  for word in message:
    if word in word_to_index:
      vector[word_to_index[word]] += 1
  return vector

for i in range(len(cleaned_data)):
  cleaned_data[i][0] = message_to_vector(cleaned_data[i][0])
  cleaned_data[i][1] = torch.tensor([cleaned_data[i][1]], dtype=torch.float32)

# split the data into training and testing data
  
random.shuffle(cleaned_data)

training_data = cleaned_data[:int(len(cleaned_data) * 0.9)]
testing_data = cleaned_data[int(len(cleaned_data) * 0.9):]

# print the number of each label in the training and testing data

spam_in_training = 0
ham_in_training = 0

for message, label in training_data:
  if label.item() == 1:
    spam_in_training += 1
  else:
    ham_in_training += 1

spam_in_testing = 0
ham_in_testing = 0

for message, label in testing_data:
  if label.item() == 1:
    spam_in_testing += 1
  else:
    ham_in_testing += 1

print("Spam in training data: ", spam_in_training)
print("Ham in training data: ", ham_in_training)
print("Spam in testing data: ", spam_in_testing)
print("Ham in testing data: ", ham_in_testing)


class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    self.fc1 = nn.Linear(len(words), 5)
    self.fc2 = nn.Linear(5, 1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.sigmoid(x)
    return x

num_epochs = 50

def train_model():
  print("training model")
  # keep track of the loss for each epoch and print it
  model = NN()
  criterion = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  for epoch in range(num_epochs):
    total_loss = 0
    for sentence, label in training_data[:1000]:
      optimizer.zero_grad()
      output = model(sentence)
      loss = criterion(output, label)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      
    print("Epoch: ", epoch + 1)
    print("Loss: ", total_loss)
  
  return model

def test_model(model):
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


model = train_model()
test_model(model)