import numpy as np
import firstGameFirstPolicyFunctions as mf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
rewardMatrix = np.array([[5, 0], [10, 1]]) ## Reward Matrix
epsilon = 0.2 ## epsilon
firstPlayerPolicy = mf.policyInitializer()
secondPlayerPolicy = mf.policyInitializer()
policyTrackPlayer1 = list()
policyTrackPlayer2 = list()
for i in range(1000):
  learningRate = 0.5/(i+1)
  policyTrackPlayer1.append(firstPlayerPolicy[0])
  policyTrackPlayer2.append(secondPlayerPolicy[0])
  firstPlayerAction = mf.actionTaker(firstPlayerPolicy, epsilon)
  secondPlayerAction = mf.actionTaker(secondPlayerPolicy, epsilon)
  firstPlayerPolicy, secondPlayerPolicy = mf.policyImprover(firstPlayerPolicy, secondPlayerPolicy, firstPlayerAction, secondPlayerAction, learningRate, rewardMatrix)

colors = cm.rainbow(np.linspace(0, 1, len(policyTrackPlayer1)))
plt.scatter(policyTrackPlayer1, policyTrackPlayer2, s=5, color=colors)
plt.scatter(firstPlayerPolicy[0], secondPlayerPolicy[0], s=50, color=(1, 0, 0))
plt.title('Prisoner\'s Dilemma')
plt.xlabel('Player 1 cooperation')
plt.ylabel('Player 2 cooperation')
plt.show()
