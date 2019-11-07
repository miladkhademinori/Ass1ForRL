import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import secondGameSecondPolicyFunctions as mf

rewardMatrix = np.array([[1, -1], [-1, 1]]) ## Reward Matrix
epsilon = 0.2 ## epsilon
firstPlayerPolicy = mf.policyInitializer()
secondPlayerPolicy = mf.policyInitializer()
firstPlayerPolicyExpect = firstPlayerPolicy
secondPlayerPolicyExpect = secondPlayerPolicy
policyTrack1Player1 = list()
policyTrack1Player2 = list()
gameValue = 0
for i in range(3000):
    learningRate = 1/(i+1)
    policyTrack1Player1.append(firstPlayerPolicy[0])
    policyTrack1Player2.append(secondPlayerPolicy[0])
    firstPlayerAction = mf.actionTaker(firstPlayerPolicy, epsilon)
    secondPlayerAction = mf.actionTaker(secondPlayerPolicy, epsilon) 
    gameValue = (gameValue * i + rewardMatrix[firstPlayerAction, secondPlayerAction])/ (i + 1)
    firstPlayerPolicy, secondPlayerPolicy = mf.policyImprover(firstPlayerPolicy, secondPlayerPolicy, firstPlayerAction, secondPlayerAction, learningRate, rewardMatrix, firstPlayerPolicyExpect, secondPlayerPolicyExpect, i)

colors = cm.rainbow(np.linspace(0, 1, len(policyTrack1Player1)))
plt.scatter(policyTrack1Player1, policyTrack1Player2, s=5, color=colors)
plt.scatter(firstPlayerPolicy[0], secondPlayerPolicy[0], s=50, color=(1, 0, 0))
plt.title('Matching Pennies')
plt.xlabel('Player 1 head')
plt.ylabel('Player 2 head')
print(gameValue)
plt.show()