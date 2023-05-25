import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pandas as pd
import numpy as np

# Define the DDQN model architecture
class DDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DDQN, self).__init__()
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
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Hyperparameters
num_epochs = 100
input_size = 38  # Length of the array
output_size = 2  # Number of actions
learning_rate = 0.001
buffer_capacity = 1000
batch_size = 256
gamma = 0.99  # Discount factor
target_update_frequency = 10

# Create the DDQN model
online_model = DDQN(input_size, output_size)
target_model = DDQN(input_size, output_size)
target_model.load_state_dict(online_model.state_dict())
target_model.eval()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(online_model.parameters(), lr=learning_rate)

# Create the replay buffer
replay_buffer = ReplayBuffer(buffer_capacity)

# Read the data
data = pd.read_csv("data/matrix1/origin/OK_http-params-normal.csv")
buffer_capacity = data.shape[0]
for i in range(len(data) - 1):
    replay_buffer.add(data.iloc[i].values[:-1], data.iloc[i].values[1],0 if data.iloc[i].values[1] == 0 else 10, data.iloc[i + 1].values[:-1], True)

eposh_losses = 0

# Training loop
for epoch in range(num_epochs):
    # Sample a mini-batch from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert the mini-batch to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    # Compute the Q-values for the current and next states using the online model
    current_q_values = online_model(states).gather(1, actions)

    # Compute the Q-values for the next states using the target model
    next_q_values = target_model(next_states)
    max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)

    # Compute the target Q-values
    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Compute the loss
    loss = criterion(current_q_values, target_q_values)

    # Perform one step of optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update the target model periodically
    if epoch % target_update_frequency == 0:
        target_model.load_state_dict(online_model.state_dict())

    # Print the current epoch
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Compute the average loss
average_loss = eposh_losses / num_epochs

# Print the total number of epochs, average loss
print(f"Total Epochs: {num_epochs}")
print(f"Average Loss: {average_loss * 100}%" )
