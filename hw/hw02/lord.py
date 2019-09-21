import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

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
