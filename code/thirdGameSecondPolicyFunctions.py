import numpy as np
def policyInitializer():
    temp = np.random.random(3)
    temp /= np.sum(temp)
    return temp

def actionTaker(inputPolicy, epsilon):
    temporary = np.argmax(inputPolicy)
    if np.random.random() > epsilon:
        return temporary
    if temporary == 0:
        if np.random.random() > 0.5:
            return 1
        else:
            return 2
    elif temporary == 1:
        if np.random.random() > 0.5:
            return 0
        else:
            return 2
    else:
        if np.random.random() > 0.5:
            return 0
        else:
            return 1

def policyImprover(inputPolicy1, inputPolicy2, inputAction1, inputAction2, learningRate, rewardMatrix, firstPlayerPolicyExpect, secondPlayerPolicyExpect, i):

    firstPlayerPolicyExpect = (firstPlayerPolicyExpect*i + inputPolicy1) / (i+1)
    secondPlayerPolicyExpect = (secondPlayerPolicyExpect*i + inputPolicy1) / (i+1)
    inputPolicy1[inputAction1] += learningRate*rewardMatrix[inputAction1][inputAction2]*(1-inputPolicy1[inputAction1]) + (learningRate/2)*(firstPlayerPolicyExpect[inputAction1] - inputPolicy1[inputAction1])
    inputPolicy1[inputAction1 - 1] += - learningRate*rewardMatrix[inputAction1][inputAction2]*inputPolicy1[inputAction1 - 1] + (learningRate/2)*(firstPlayerPolicyExpect[inputAction1 - 1] - inputPolicy1[inputAction1 - 1])
    inputPolicy1[inputAction1 - 2] += - learningRate*rewardMatrix[inputAction1][inputAction2]*inputPolicy1[inputAction1 - 2] + (learningRate/2)*(firstPlayerPolicyExpect[inputAction1 - 2] - inputPolicy1[inputAction1 - 2])
    inputPolicy2[inputAction2] += - learningRate*rewardMatrix[inputAction1][inputAction2]*(1-inputPolicy2[inputAction2]) + (learningRate/2)*(secondPlayerPolicyExpect[inputAction2] - inputPolicy2[inputAction2])
    inputPolicy2[inputAction2 - 1] += + learningRate*rewardMatrix[inputAction1][inputAction2]*inputPolicy2[inputAction2 -1] + (learningRate/2)*(secondPlayerPolicyExpect[inputAction2 -1] - inputPolicy2[inputAction2 -1])
    inputPolicy2[inputAction2 - 2] += + learningRate*rewardMatrix[inputAction1][inputAction2]*inputPolicy2[inputAction2 -2] + (learningRate/2)*(secondPlayerPolicyExpect[inputAction2 -2] - inputPolicy2[inputAction2 -2])
    inputPolicy1, inputPolicy2 = np.fmax(inputPolicy1, 0), np.fmax(inputPolicy2, 0)
    inputPolicy1 /= np.sum(inputPolicy1)
    inputPolicy2 /= np.sum(inputPolicy2)
    return inputPolicy1, inputPolicy2
