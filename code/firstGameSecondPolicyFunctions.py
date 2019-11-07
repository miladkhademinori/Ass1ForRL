import numpy as np
def policyInitializer():
    temp = np.random.random([2])
    temp /= np.sum(temp)
    return temp

def actionTaker(inputPolicy, epsilon):
    return np.argmax(inputPolicy) if np.random.random() > epsilon else (1 if np.argmax(inputPolicy) == 0 else 0) 

def policyImprover(inputPolicy1, inputPolicy2, inputAction1, inputAction2, learningRate, rewardMatrix, firstPlayerPolicyExpect, secondPlayerPolicyExpect, i):
    firstPlayerPolicyExpect = (firstPlayerPolicyExpect*i + inputPolicy1) / (i+1)
    secondPlayerPolicyExpect = (secondPlayerPolicyExpect*i + inputPolicy1) / (i+1)
    inputPolicy1[inputAction1] += learningRate*rewardMatrix[inputAction1][inputAction2]*(1-inputPolicy1[inputAction1])+(learningRate/2)*(firstPlayerPolicyExpect[inputAction1] - inputPolicy1[inputAction1])
    inputPolicy1[1-inputAction1] += -learningRate*rewardMatrix[inputAction1][inputAction2]*inputPolicy1[1-inputAction1]+(learningRate/2)*(firstPlayerPolicyExpect[inputAction1 - 1] - inputPolicy1[inputAction1 - 1])
    inputPolicy2[inputAction2] += learningRate*np.transpose(rewardMatrix)[inputAction1][inputAction2]*(1-inputPolicy2[inputAction2])+ (learningRate/2)*(secondPlayerPolicyExpect[inputAction2] - inputPolicy2[inputAction2])
    inputPolicy2[1-inputAction2] += - learningRate*np.transpose(rewardMatrix)[inputAction1][inputAction2]*inputPolicy2[1-inputAction2]+ (learningRate/2)*(secondPlayerPolicyExpect[inputAction2 -1] - inputPolicy2[inputAction2 -1])
    inputPolicy1, inputPolicy2 = np.fmax(inputPolicy1, 0), np.fmax(inputPolicy2, 0)
    inputPolicy1 /= np.sum(inputPolicy1)
    inputPolicy2 /= np.sum(inputPolicy2)
    return inputPolicy1, inputPolicy2

