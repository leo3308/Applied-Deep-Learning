import os
import json
import matplotlib.pyplot as plt

with open('policy.json', 'r') as f:
    file = json.load(f)

x1 = file['pg']['x']
y1 = file['pg']['y']
x2 = file['pg_baseline']['x']
y2 = file['pg_baseline']['y']

title = 'policy gradient'
plt.plot(x1, y1, linewidth=3, label='w/o baseline')
plt.plot(x2, y2, linewidth=3, label='w/ baseline')
plt.title(title, fontsize=14)
plt.xlabel("Steps", fontsize=10)
plt.ylabel("Avg Reward", fontsize=10)
plt.legend()
plt.savefig('policy_gradient.png')

