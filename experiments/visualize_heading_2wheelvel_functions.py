"""
Script that visualizes how a scalar action (steering value) is mapped to wheel velocity signals that can control a
Duckiebot, which is a differential-wheeled robot.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
import matplotlib.pyplot as plt
import numpy as np

speed=1
action = np.linspace(-1., 1., 100)
# wheelVels = [-0.5 * action + 0.5, 0.5 * action + 0.5]
wheelVels = np.clip([1 - action, 1 + action], 0., 1)
# wheelVels = np.clip([1 - action*0.6666, 1 + action*0.6666], 0., 1)
# wheelVels = np.clip([1 - np.sin(action*3.14), 1 + np.sin(action*3.14)], 0., 1)
# wheelVels = np.clip([1 - action**3, 1 + action**3], 0., 1)

# straight_plateau_width = 0.3333
# mul = 1./(1.-straight_plateau_width)
# wheelVels = np.clip(np.array([1 - action, 1 + action])*mul, 0., 1)

plt.plot(np.stack([action, action], axis=1), np.transpose(wheelVels))
plt.xticks([-1., 0., 1.])
plt.yticks([0., 0.5, 1.])
plt.xlabel("Scalar value")
plt.ylabel("Wheel velocities")
plt.legend(["Left wheel speed", "Right wheel speed"])
plt.tight_layout()
plt.show()

plt.fill_betweenx([-1, 1], [1, 1], [-1, -1], alpha=0.25, color='gray')
plt.scatter(np.transpose(wheelVels)[:, 0], np.transpose(wheelVels)[:, 1], c=action)
cbar = plt.colorbar()
cbar.set_label('Scalar value', rotation=90)
cbar.set_ticks([-1., 0., 1.])
plt.xticks([-1., 0., 1.])
plt.yticks([-1., 0., 1.])
plt.xlabel("Left wheel speed")
plt.ylabel("Right wheel speed")
# plt.legend(["Original action space"], bbox_to_anchor=(-0.05, 1.05), loc='lower left' )
plt.legend(["Original action space"], bbox_to_anchor=(0.05, 0.05), loc='lower left')
plt.axis('equal')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()
