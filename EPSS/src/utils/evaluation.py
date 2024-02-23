import sklearn
import numpy as np

def eval_acc(Y, P):
    correct = 0
    for y, p in zip(Y, P):
        if y == p:
            correct += 1
    return correct / len(Y)

def eval_mse(Y, P, config):
    correct = 0
    for y, p in zip(Y, P):
        correct += np.power((y/config.horizon-p/config.horizon),2)
    return correct / len(Y)

def eval_r2(Y, P, config):
    correct = 0
    for y, p in zip(Y, P):
        correct += np.power((y/config.horizon-p/config.horizon),2)
    mse =  correct / len(Y)
    return 1 - mse/np.var([y/config.horizon for y in Y])

def get_rates(true, probs, threshold):
    TP, FP = 0, 0
    positives = [x for x in true if x == 1]
    for t, p in zip(true, probs):
        if p>= threshold:
            if t == 1:
                TP+=1
            else:
                FP+=1
    return TP/len(positives), FP/(len(true)-len(positives))

def eval_auc(true, epss_scores):
    thresholds = np.linspace(0, 1, num=1000, endpoint=True)
    tps, fps = [], []
    for t in thresholds:
        rates = get_rates(true, epss_scores, t)
        tps.append(rates[0])
        fps.append(rates[1])
    return sklearn.metrics.auc(fps, tps)
