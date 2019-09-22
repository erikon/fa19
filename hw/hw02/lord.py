import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import math


def calculate_p_values(x, mean_0, stdev_0):
    # for a one sided test, with h_1: mu > mean_0
    z_scores = (x - mean_0)/stdev_0
    return 1 - scipy.stats.norm.cdf(z_scores)

def question1c_all():
    # p-values = stats.norm.cdf(data, mu, sigma)
    def question1c():
        true_values = []
        samples = []
        mu = []
        for i in range(0, 501):
            if i >= 0 and i <= 450:
                mu.append(0)
                true_values.append(1)
            else:
                mu.append((i-450)/5)
                true_values.append(0)
            samples.append(np.random.normal(mu[i], 1, 1)[0])

        p_values = stats.norm.cdf(samples, 0, 1)
        # print(true_values)
        # print(p_values)

        return true_values, p_values, samples

    alpha = [.001, .005, .01, .05, .1, .2, .3, .4, .5]
    avg_reject = []
    avg_true_reject = []
    avg_false_rate = []
    for a in alpha:
        rejects = []
        t_rejects = []
        fdr = []
        for i in range(100):
            tvals, pvals, samples = question1c()
            # print(tvals)
            # print(pvals)

            # p <= a
            # p <= a/N

            # temp_array = np.sort(pvals)
            # largest = 0
            # for k in range(len(temp_array)):
            #     if temp_array[k] <= k * a/500:
            #         largest = temp_array[k]
            # decisions = [int(p <= largest) for p in pvals]

            # a_weighted = a * 2 * i / 501
            # a_weighted = a * 2 * (501 - i)/501

            w = 0.5 if (i >= 0 and i <= 450) else 5.5
            a_weighted = a * w / 501

            decisions = [int(p <= a_weighted) for p in pvals]

            rejects.append(sum([1 if d==1 else 0 for d in decisions]))
            t_rejects.append(sum([1 if (decisions[i] == 0 and tvals[i] == 0) else 0 for i in range(len(decisions))]))
            FP_count = sum([1 if (decisions[i] == 1 and tvals[i] == 0) else 0 for i in range(len(decisions))])
            TP_count = sum([1 if (decisions[i] == 1 and tvals[i] == 1) else 0 for i in range(len(decisions))])
            if (FP_count + TP_count) == 0:
                fdr.append(0)
            else:
                fdr.append(FP_count / (FP_count + TP_count))

        # print(t_rejects)
        # print(fdr)
        avg_reject.append(sum(rejects)/100.0)
        avg_true_reject.append(sum(t_rejects)/100.0)
        avg_false_rate.append(sum(fdr)/100.0)

    plt.figure()
    plt.plot(alpha, avg_reject)
    plt.plot(alpha, avg_true_reject)
    plt.plot(alpha, avg_false_rate)
    print(avg_reject)
    print(avg_true_reject)
    print(avg_false_rate)
    plt.show()

def LORD(stream,alpha):
    # Inputs: stream - array of p-values, alpha - target FDR level
    # Output: array of indices k such that the k-th p-value corresponds to a discovery

    gamma = lambda t: 6/(math.pi*t)**2
    w_0 = alpha/2
    n = len(stream)
    rejections = []
    alpha_t = gamma(1)*w_0
    for t in range(1,n+1):
        # Offset by one since indexing by 1 for t
        p_t = stream[t-1]

        if p_t < alpha_t:
            rejections.append(t)

        next_alpha_t = gamma(t+1)*w_0 + alpha*sum([gamma(t+1-rej) for rej in rejections])
        # Check if tau_1 exists
        if len(rejections)>0:
            next_alpha_t -= gamma(t+1-rejections[0])*w_0


        # Update alpha
        alpha_t = next_alpha_t
    # Shift rejections since the rejections are 1-indexed
    shifted_rej = [rej-1 for rej in rejections]
    return shifted_rej



pis = [0.1, 0.3, 0.5, 0.7, 0.9]
N = 1000
alpha = 0.5
def gen_pval_1(pi):
    pvals = []
    thetas = np.random.binomial(size=N, n=1, p=1-pi)
    for theta in thetas:
        if theta == 0:
            pvals.append(np.random.uniform(low=0, high=1, size=1)[0])
        else:
            samples = np.random.normal(loc=3, scale=1, size=N)
            pvals.append(stats.norm.cdf(samples, 0, 1)[0])
    return pvals, thetas

def gen_pval_2(pi):
    pvals = [0]*N
    thetas = [0]*N
    piN = int(pi*N)
    for i in range(piN):
        thetas[i] = 0
        pvals[i] = np.random.uniform(low=0, high=1, size=1)[0]

    for i in range(piN + 1, N):
        thetas[i] = 1
        samples = np.random.normal(loc=3, scale=1, size=N)
        pvals[i] = stats.norm.cdf(samples, 0, 1)[0]

    return pvals, thetas

def gen_pval_3(pi):
    pvals = [0]*N
    thetas = [0]*N

    for i in range(int(N - pi*N)):
        thetas[i] = 1
        samples = np.random.normal(loc=3, scale=1, size=N)
        pvals[i] = stats.norm.cdf(samples, 0, 1)[0]

    for i in range(int(N - pi*N + 1), N):
        thetas[i] = 0
        pvals[i] = np.random.uniform(low=0, high=1, size=1)[0]

    return pvals, thetas

def BH(pvals, alpha):
    temp_array = np.sort(pvals)
    largest = 0
    for k in range(len(temp_array)):
        if temp_array[k] <= k * alpha/500:
            largest = temp_array[k]
    decisions = [int(p <= largest) for p in pvals]
    return decisions

total_false_gen1 = []
total_sense_gen1 = []

total_false_gen2 = []
total_sense_gen2 = []

total_false_gen3 = []
total_sense_gen3 = []

for index in range(3):
    avg_false_rate = []
    avg_sense = []
    for pi in pis:
        fdp_s = []
        sense_s = []
        for i in range(100):
            if index == 0:
                pvals, thetas = gen_pval_1(pi)
            elif index == 1:
                pvals, thetas = gen_pval_2(pi)
            else:
                pvals, thetas = gen_pval_3(pi)

            # discoveries = LORD(stream=pvals, alpha=alpha)

            discoveries = BH(pvals, alpha)


            FN_count = sum([1 if (discoveries[j] == 0 and thetas[j] == 1) else 0 for j in range(len(discoveries))])
            TP_count = sum([1 if (discoveries[j] == 1 and thetas[j] == 1) else 0 for j in range(len(discoveries))])
            FP_count = sum([1 if (discoveries[j] == 1 and thetas[j] == 0) else 0 for j in range(len(discoveries))])

            if (FP_count + TP_count) == 0:
                fdp = 0
            else:
                fdp = float(FP_count / (FP_count + TP_count))

            if (FN_count + TP_count) == 0:
                sense = 0
            else:
                sense = float(TP_count / (TP_count + FN_count))

            fdp_s.append(fdp)
            sense_s.append(sense)
        avg_false_rate.append(sum(fdp_s)/100.0)
        avg_sense.append(sum(sense_s)/100.0)

    if index == 0:
        total_false_gen1.append(avg_false_rate)
        total_sense_gen1.append(avg_sense)
    elif index == 1:
        total_false_gen2.append(avg_false_rate)
        total_sense_gen2.append(avg_sense)
    else:
        total_false_gen3.append(avg_false_rate)
        total_sense_gen3.append(avg_sense)

print(total_false_gen1, type(total_false_gen1))
print(total_false_gen2)
print(total_false_gen3)

print(total_sense_gen1)
print(total_sense_gen2)
print(total_sense_gen3)

plt.figure()
plt.plot(pis, total_false_gen1[0])
plt.plot(pis, total_false_gen2[0])
plt.plot(pis, total_false_gen3[0])
plt.show()

plt.figure()
plt.plot(pis, total_sense_gen1[0])
plt.plot(pis, total_sense_gen2[0])
plt.plot(pis, total_sense_gen3[0])
plt.show()
