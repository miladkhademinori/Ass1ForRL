import numpy as np
import thirdGameSecondPolicyFunctions as mf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
rewardMatrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]) ## Reward Matrix
epsilon = 0.15 ## epsilon
firstPlayerPolicy = mf.policyInitializer()
secondPlayerPolicy = mf.policyInitializer()
firstPlayerPolicyExpect = firstPlayerPolicy
secondPlayerPolicyExpect = secondPlayerPolicy
policyTrackPlayer11 = list()
policyTrackPlayer12 = list()
policyTrackPlayer13 = list()
gameValue = 0
for i in range(30000):
  learningRate = 0.3/(i+1)**(0.5)    #learningRate = 4/(i+1)
  policyTrackPlayer11.append(firstPlayerPolicy[0])
  policyTrackPlayer12.append(firstPlayerPolicy[1])
  policyTrackPlayer13.append(firstPlayerPolicy[2])
  firstPlayerAction = mf.actionTaker(firstPlayerPolicy, epsilon)
  secondPlayerAction = mf.actionTaker(secondPlayerPolicy, epsilon)
  gameValue = (gameValue * i + rewardMatrix[firstPlayerAction, secondPlayerAction])/ (i + 1)
  firstPlayerPolicy, secondPlayerPolicy = mf.policyImprover(firstPlayerPolicy, secondPlayerPolicy, firstPlayerAction, secondPlayerAction, learningRate, rewardMatrix, firstPlayerPolicyExpect, secondPlayerPolicyExpect, i)

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
print(gameValue)
plt.show()
