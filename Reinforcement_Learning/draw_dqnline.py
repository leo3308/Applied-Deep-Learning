import os
import json
import matplotlib.pyplot as plt

with open('all_curve.json', 'r') as f:
    file = json.load(f)

x1 = file['gamma0.99']['x']
y1 = file['gamma0.99']['y']
x2 = file['gamma1']['x']
y2 = file['gamma1']['y']
x3 = file['gamma0.75']['x']
y3 = file['gamma0.75']['y']
x4 = file['gamma0.50']['x']
y4 = file['gamma0.50']['y']

title = 'different gamma curve'
plt.plot(x1, y1, linewidth=3, label='GAMMA 0.99')
plt.plot(x2, y2, linewidth=3, label='GAMMA 1.00')
plt.plot(x3, y3, linewidth=3, label='GAMMA 0.75')
plt.plot(x4, y4, linewidth=3, label='GAMMA 0.50')
plt.title(title, fontsize=14)
plt.xlabel("Steps", fontsize=10)
plt.ylabel("Avg Reward", fontsize=10)
plt.legend()
plt.savefig('gamma_curve.png')

