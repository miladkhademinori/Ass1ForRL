import numpy as np
import firstGameSecondPolicyFunctions as mf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
rewardMatrix = np.array([[5, 0], [10, 1]]) ## Reward Matrix
epsilon = 0.1 ## epsilon
firstPlayerPolicy = mf.policyInitializer()
secondPlayerPolicy = mf.policyInitializer()
firstPlayerPolicyExpect = firstPlayerPolicy
secondPlayerPolicyExpect = secondPlayerPolicy
policyTrackPlayer1 = list()
policyTrackPlayer2 = list()
gameValue = 0

for i in range(30000):
  learningRate = 0.5/(i+1)**(0.6)
  policyTrackPlayer1.append(firstPlayerPolicy[0])
  policyTrackPlayer2.append(secondPlayerPolicy[0])
  firstPlayerAction = mf.actionTaker(firstPlayerPolicy, epsilon)
  secondPlayerAction = mf.actionTaker(secondPlayerPolicy, epsilon)
  gameValue = (gameValue * i + rewardMatrix[firstPlayerAction, secondPlayerAction])/ (i + 1)
  firstPlayerPolicy, secondPlayerPolicy = mf.policyImprover(firstPlayerPolicy, secondPlayerPolicy, firstPlayerAction, secondPlayerAction, learningRate, rewardMatrix, firstPlayerPolicyExpect, secondPlayerPolicyExpect, i)

colors = cm.rainbow(np.linspace(0, 1, len(policyTrackPlayer1)))
plt.scatter(policyTrackPlayer1, policyTrackPlayer2, s=5, color=colors)
plt.scatter(firstPlayerPolicy[0], secondPlayerPolicy[0], s=50, color=(1, 0, 0))
plt.title('Prisoners\' Dilemma')
plt.xlabel('Player 1 cooperation')
plt.ylabel('Player 2 cooperation')
print(gameValue)
plt.show()
