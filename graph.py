import numpy as np
from matplotlib import pyplot as plt

# find_and_correct = [(902, 641), (903, 651), (704, 657), (704, 669)]
# scores = [(0.7106, 0.8368, 0.7685), (0.7209, 0.8498, 0.7801,), (0.9332, 0.8577, 0.8938), (0.9502, 0.8733, 0.9102,)]

plt.rcParams['font.sans-serif'] = 'SimHei'

labels = ['以上一个时间为基准',
          '以全局时间为基准',
          '本文方法']
p = [0.7737,
     0.9632,
     0.9675]
r = [0.7141,
     0.8890,
     0.8930]
f1 = [0.7427,
      0.9246,
      0.9287]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, p, width, label='P', color='#45b97c')
rects2 = ax.bar(x, r, width, label='R', color='#f05b72')
rects3 = ax.bar(x + width, f1, width, label='F-score', color='#426ab3')

ax.set_ylabel('Scores')
ax.set_title('对比实验3')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(framealpha=0.45)

padding = 3
ax.bar_label(rects1, padding=padding)
ax.bar_label(rects2, padding=padding)
ax.bar_label(rects3, padding=padding)
fig.tight_layout()

plt.ylim(0, 1.1)
plt.show()
