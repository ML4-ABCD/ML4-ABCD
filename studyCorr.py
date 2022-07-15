import math
import random
import numpy as np
import matplotlib.pyplot as plt
from Correlation import Correlation as corr

def main():
    ###########################
    #Define simulation constants
    ###########################
    nEvents = 1000
    sigma = 0.005

    ###########################
    # Simulate events, fill background and signal events. Need to optimize
    ###########################
    a1 = np.random.multivariate_normal([0.1,0.1], [[sigma, 0],[0, sigma]], 1*nEvents)
    a2 = np.random.multivariate_normal([0.9,0.1], [[sigma, 0],[0, sigma]], 1*nEvents)
    a3 = np.random.multivariate_normal([0.1,0.9], [[sigma, 0],[0, sigma]], 1*nEvents)
    a4 = np.random.multivariate_normal([0.9,0.9], [[sigma, 0],[0, sigma]], 1*nEvents)
    a = np.concatenate((a1,a2,a3,a4))
    a = a[~np.any(a<0.0, axis=1)]
    d1Tot, d2Tot = a.T

    print("Distance correlation:", corr.distance_corr(d1Tot, d2Tot, np.ones_like(d1Tot)).numpy())
    print("Pearson  correlation:", corr.pearson_corr_tf(d1Tot, d2Tot).numpy())
    print("RDC                 :", corr.rdc(d1Tot, d2Tot))

    fig = plt.figure()
    plt.hist2d(d1Tot, d2Tot, bins=100)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    fig.savefig("trainVar.png", dpi=fig.dpi)

if __name__ == '__main__':
    main()
