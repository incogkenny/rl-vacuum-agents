import matplotlib.pyplot as plt
import numpy as np

# Data
agents = ['PPO', 'PPO', 'PPO', 'DQN', 'DQN', 'DQN']
trained_on = ['Layout 1', 'Layout 1', 'Layout 1', 'Layout 1', 'Layout 1', 'Layout 1']
tested_on = ['Layout 1', 'Layout 2', 'Layout 3', 'Layout 1', 'Layout 2', 'Layout 3']
dirt_cleaned = [91.4, 80.5, 83.0, 80.6, 58.0, 64.5]
performance_retention = [100, 88, 91, 100, 72, 80]

# Grouped data
layouts = ['Layout 1', 'Layout 2', 'Layout 3']
ppo_dirt_cleaned = dirt_cleaned[:3]
ppo_performance_retention = performance_retention[:3]
dqn_dirt_cleaned = dirt_cleaned[3:]
dqn_performance_retention = performance_retention[3:]

# Bar width and positions
bar_width = 0.35
x = np.arange(len(layouts))

# Plot
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Dirt Cleaned
ax[0].bar(x - bar_width / 2, ppo_dirt_cleaned, bar_width, label='PPO', color='blue')
ax[0].bar(x + bar_width / 2, dqn_dirt_cleaned, bar_width, label='DQN', color='orange')
ax[0].set_title('Dirt Cleaned (%)')
ax[0].set_ylabel('Percentage')
ax[0].set_xticks(x)
ax[0].set_xticklabels(layouts)
ax[0].legend()
ax[0].grid(axis='y', linestyle='--', alpha=0.7)

# Performance Retention
ax[1].bar(x - bar_width / 2, ppo_performance_retention, bar_width, label='PPO', color='blue')
ax[1].bar(x + bar_width / 2, dqn_performance_retention, bar_width, label='DQN', color='orange')
ax[1].set_title('Performance Retention (%)')
ax[1].set_ylabel('Percentage')
ax[1].set_xticks(x)
ax[1].set_xticklabels(layouts)
ax[1].legend()
ax[1].grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.show()