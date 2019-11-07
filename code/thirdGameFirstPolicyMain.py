import numpy as np
import thirdGameFirstPolicyFunctions as mf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
rewardMatrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]) ## Reward Matrix
epsilon = 0.1 ## epsilon
firstPlayerPolicy = mf.policyInitializer()
secondPlayerPolicy = mf.policyInitializer()
policyTrackPlayer11 = list()
policyTrackPlayer12 = list()
policyTrackPlayer13 = list()
for i in range(70000):
  learningRate = 0.1/(i+1)**(0.55)
  policyTrackPlayer11.append(firstPlayerPolicy[0])
  policyTrackPlayer12.append(firstPlayerPolicy[1])
  policyTrackPlayer13.append(firstPlayerPolicy[2])
  firstPlayerAction = mf.actionTaker(firstPlayerPolicy, epsilon)
  secondPlayerAction = mf.actionTaker(secondPlayerPolicy, epsilon)  
  firstPlayerPolicy, secondPlayerPolicy = mf.policyImprover(firstPlayerPolicy, secondPlayerPolicy, firstPlayerAction, secondPlayerAction, learningRate, rewardMatrix)
  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = cm.rainbow(np.linspace(0, 1, len(policyTrackPlayer11)))
ax.scatter(policyTrackPlayer11, policyTrackPlayer12, policyTrackPlayer13, s=5, color=colors)
ax.scatter(firstPlayerPolicy[0], firstPlayerPolicy[1], firstPlayerPolicy[2], s=50, color=(1, 0, 0))
print(firstPlayerPolicy)
ax.set_title("Rock-paper-scissors")
ax.set_xlabel('Rock')
ax.set_ylabel('Paper')
ax.set_zlabel('Scissors')
plt.show()
