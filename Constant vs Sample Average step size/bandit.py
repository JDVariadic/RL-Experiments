import numpy as np
import random

class TestBed:
    def __init__(self, bandits, step_size, epsilon, step_distribution):
        self.bandits = bandits
        self.step_size = step_size
        self.epsilon = epsilon
        self.step_distribution = step_distribution
        self.rewards = [0] * len(bandits)
    
    def get_bandit(self, idx):
        return self.bandits[idx]
    
    def get_bandits(self):
        return self.bandits
    
    def take_step(self):
        #Change mean reward of bandit's attribute and not bandit itself
        for bdt in self.bandits:
            bdt.step_mean_reward(self.step_distribution)

class Bandit:
    def __init__(self):
        self.mean_reward = random.uniform(0, 1)
        self.standard_distrubtion = 1

    def sample_data(self):
        return np.random.normal(self.mean_reward, self.standard_distrubtion)

    def get_mean_reward(self):
        return self.mean_reward
    
    def step_mean_reward(self, step_size):
        self.mean_reward += np.random.normal(0, 0.01)
    
class TraditionalAgent:
    def __init__(self, num_bandits, step_size, epsilon):
        self.step_size = step_size
        self.epsilon = epsilon
        self.rewards = [0] * num_bandits
        self.num_samples = [0] * num_bandits
        self.total_reward = []

    def sample_bandit(self, bandits): #Move to Agent class
        if random.random() < self.epsilon:
            chosen_idx = np.random.randint(len(bandits))
            chosen_bandit = bandits[chosen_idx]
            return [chosen_idx, chosen_bandit.sample_data()]
        else:
            chosen_idx = np.argmax(self.rewards)
            chosen_bandit = bandits[chosen_idx]
            return [chosen_idx, chosen_bandit.sample_data()]

    def update_reward_system(self, idx, reward): #Move to Agent class
        self.num_samples[idx] += 1
        self.rewards[idx] += (1/self.num_samples[idx]) * (reward - self.rewards[idx])

    def get_rewards(self):
        return self.rewards
    
    def add_to_total_reward(self, reward):
        self.total_reward.append(reward)
    
    def get_accummulated_rewards(self):
        return self.total_reward

class ConstantStepSizeAgent:
    def __init__(self, num_bandits, step_size, epsilon):
        self.step_size = step_size
        self.epsilon = epsilon
        self.rewards = [0] * num_bandits
        self.total_reward = []

    def sample_bandit(self, bandits): #Move to Agent class
        if random.random() < self.epsilon:
            chosen_idx = np.random.randint(len(bandits))
            chosen_bandit = bandits[chosen_idx]
            return [chosen_idx, chosen_bandit.sample_data()]
        else:
            chosen_idx = np.argmax(self.rewards)
            chosen_bandit = bandits[chosen_idx]
            return [chosen_idx, chosen_bandit.sample_data()]

    def update_reward_system(self, idx, reward): #Move to Agent class
        self.rewards[idx] += self.step_size * (reward - self.rewards[idx])

    def get_rewards(self):
        return self.rewards
    
    def add_to_total_reward(self, reward):
        self.total_reward.append(reward)

    def get_accummulated_rewards(self):
        return self.total_reward



