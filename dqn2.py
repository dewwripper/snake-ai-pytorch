import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pandas as pd
import numpy as np

# Define the DQN model architecture
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        print (self.buffer)
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Hyperparameters
input_size = 38  # Length of the array
output_size = 2  # Number of actions
learning_rate = 0.001
batch_size = 256
gamma = 0.9  # Discount factor
buffer_capacity = 1000
episode = 20

# Create the DQN model
model = DQN(input_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the replay buffer
replay_buffer = ReplayBuffer(buffer_capacity)

# Read the data
data = pd.read_csv("data/matrix1/origin/OK_http-params-normal.csv")
buffer_capacity = data.shape[0]
for i in range(len(data) - 1):
    replay_buffer.add(data.iloc[i].values[:-1], data.iloc[i].values[1],0 if data.iloc[i].values[1] == 0 else 10, data.iloc[i + 1].values[:-1], True)


eposh_losses = 0
# Training loop
for epoch in range(episode):
    # Sample a mini-batch from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    # states = np.array(data.iloc[epoch].values[:-1], dtype=int)
    # actions = np.array([0,1], dtype=int)
    # rewards = np.array([0,1], dtype=int)
    # next_states = np.array(data.iloc[epoch + 1].values[:-1], dtype=int)
    # dones = [True]
    

    # Convert the mini-batch to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    # Compute the Q-values for the current and next states
    current_q_values = model(states).gather(1, actions)
    next_q_values = model(next_states).max(1)[0].unsqueeze(1)

    # Compute the target Q-values
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute the loss
    loss = criterion(current_q_values, target_q_values)

    # Perform one step of optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    eposh_losses = eposh_losses + loss.item()

    # Print the loss
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


# Compute the average loss
average_loss = eposh_losses / episode

# Print the total number of epochs, average loss
print(f"Total Epochs: {episode}")
print(f"Average Loss: {average_loss * 100}%" )
