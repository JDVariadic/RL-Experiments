import numpy as np
import bandit

import matplotlib.pyplot as plt

NUM_OF_ITERATIONS = 100000
NUM_OF_BANDITS = 10
STEP_SIZE = 0.1
EPSILON = 0.1
test_bandits = list()

for i in range(NUM_OF_BANDITS):
    new_bandit = bandit.Bandit()
    test_bandits.append(new_bandit)

testbed = bandit.TestBed(test_bandits, STEP_SIZE, EPSILON, np.random.normal())
bandits = testbed.get_bandits()
sample_average_agent = bandit.TraditionalAgent(NUM_OF_BANDITS, STEP_SIZE, EPSILON)
constant_step_size_agent = bandit.ConstantStepSizeAgent(NUM_OF_BANDITS, STEP_SIZE, EPSILON)

for i in range(NUM_OF_ITERATIONS):

    sample_chosen_bandit, sample_reward_value = sample_average_agent.sample_bandit(bandits)
    constant_chosen_bandit, constant_reward_value = constant_step_size_agent.sample_bandit(bandits)

    sample_average_agent.add_to_total_reward(sample_reward_value)
    constant_step_size_agent.add_to_total_reward(constant_reward_value)

    sample_average_agent.update_reward_system(sample_chosen_bandit, sample_reward_value)
    constant_step_size_agent.update_reward_system(constant_chosen_bandit, constant_reward_value)

    testbed.take_step()

sample_step_size_rewards = sample_average_agent.get_accummulated_rewards()
constant_step_size_rewards = constant_step_size_agent.get_accummulated_rewards()
print("Sampled step size rewards: ", sum(sample_step_size_rewards))
print("Constant step size rewards: ", sum(constant_step_size_rewards))
average_sample_step_size_rewards = []
average_constant_step_size_rewards = []
chunk_size = 1000

for i in range(0, len(sample_step_size_rewards), chunk_size):
    sample_chunk = sample_step_size_rewards[i:i + chunk_size]
    constant_chunk = constant_step_size_rewards[i:i + chunk_size]

    average_sample_step_size_rewards.append(sum(sample_chunk) / len(sample_chunk))
    average_constant_step_size_rewards.append(sum(constant_chunk) / len(constant_chunk))

iteration_range = np.arange(0, len(sample_step_size_rewards), chunk_size)

plt.figure(figsize=(15, 6))

# First subplot for Sampled Step Size Rewards
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(iteration_range, average_sample_step_size_rewards, label='Sample Average Method Rewards', color='blue')
plt.title("Average Sampled Step Size Rewards per 100 Episodes")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()

# Second subplot for Constant Step Size Rewards
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(iteration_range, average_constant_step_size_rewards, label='Constant Step Size Rewards', color='red')
plt.title("Average Constant Step Size Rewards per 100 Episodes")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the figure with both subplots
plt.show()
