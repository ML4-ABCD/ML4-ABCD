import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata

class Correlation:
    @staticmethod
    def distance_corr(var_1, var_2, normedweight, power=1):
        """
        https://github.com/gkasieczka/DisCo
        var_1: First variable to decorrelate (eg mass)
        var_2: Second variable to decorrelate (eg classifier output)
        normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
        power: Exponent used in calculating the distance correlation
        
        va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries
        
        Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
        """
        
        xx = tf.reshape(var_1, [-1, 1])
        xx = tf.tile(xx, [1, tf.size(var_1)])
        xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])
        
        yy = tf.transpose(xx)
        amat = tf.abs(xx-yy)
        
        xx = tf.reshape(var_2, [-1, 1])
        xx = tf.tile(xx, [1, tf.size(var_2)])
        xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
        
        yy = tf.transpose(xx)
        bmat = tf.abs(xx-yy)
        
        amatavg = tf.reduce_mean(amat*normedweight, axis=1)
        bmatavg = tf.reduce_mean(bmat*normedweight, axis=1)
        
        minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
        minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
        minuend_2 = tf.transpose(minuend_1)
        Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg*normedweight)
        
        minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
        minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
        minuend_2 = tf.transpose(minuend_1)
        Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg*normedweight)
        
        ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
        AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
        BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)
        
        if power==1:
            dCorr = tf.reduce_mean(ABavg*normedweight)/tf.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
        elif power==2:
            dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
        else:
            dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)))**power
        return dCorr
            
    def dist_corr(X, Y):
        """ 
        https://gist.github.com/satra/aa3d19a12b74e9ab7941
        Compute the distance correlation function
        
        >>> a = [1,2,3,4,5]
        >>> b = np.array([1,2,9,4,4])
        >>> distcorr(a, b)
        0.762676242417
        """
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        if np.prod(X.shape) == len(X):
            X = X[:, None]
        if np.prod(Y.shape) == len(Y):
            Y = Y[:, None]
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        n = X.shape[0]
        if Y.shape[0] != X.shape[0]:
            raise ValueError('Number of samples must match')
        a = squareform(pdist(X))
        b = squareform(pdist(Y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcor

    def pearson_corr_tf(v, u):
        v = tf.reshape(v, [-1, 1])
        u = tf.reshape(u, [-1, 1])

        mv, sv = tf.nn.moments(v, axes=[0])
        mu, su = tf.nn.moments(u, axes=[0])

        ev = v - mv
        eu = u - mu

        j = ev*eu
        mj = tf.reduce_mean(j)

        num = mj
        den = sv*su

        return tf.abs(num)

    def pearson_corr(X, Y):
        cor, pvalue = pearsonr(X, Y)
        return cor

    def spearman_corr(X, Y):
        cor, pvalue = spearmanr(X, Y)
        return cor

    """
    Implements the Randomized Dependence Coefficient
    David Lopez-Paz, Philipp Hennig, Bernhard Schoelkopf

    http://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf
    """
    def rdc(x, y, f=np.sin, k=20, s=1/6., n=1):
        """
        Computes the Randomized Dependence Coefficient
        x,y: numpy arrays 1-D or 2-D
             If 1-D, size (samples,)
             If 2-D, size (samples, variables)
        f:   function to use for random projection
        k:   number of random projections to use
        s:   scale parameter
        n:   number of times to compute the RDC and
             return the median (for stability)
    
        According to the paper, the coefficient should be relatively insensitive to
        the settings of the f, k, and s parameters.
        """
        if n > 1:
            values = []
            for i in range(n):
                try:
                    values.append(rdc(x, y, f, k, s, 1))
                except np.linalg.linalg.LinAlgError: pass
            return np.median(values)
    
        if len(x.shape) == 1: x = x.reshape((-1, 1))
        if len(y.shape) == 1: y = y.reshape((-1, 1))
    
        # Copula Transformation
        cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
        cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)
    
        # Add a vector of ones so that w.x + b is just a dot product
        O = np.ones(cx.shape[0])
        X = np.column_stack([cx, O])
        Y = np.column_stack([cy, O])
    
        # Random linear projections
        Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
        Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
        X = np.dot(X, Rx)
        Y = np.dot(Y, Ry)
    
        # Apply non-linear function to random projections
        fX = f(X)
        fY = f(Y)
    
        # Compute full covariance matrix
        C = np.cov(np.hstack([fX, fY]).T)
    
        # Due to numerical issues, if k is too large,
        # then rank(fX) < k or rank(fY) < k, so we need
        # to find the largest k such that the eigenvalues
        # (canonical correlations) are real-valued
        k0 = k
        lb = 1
        ub = k
        while True:
    
            # Compute canonical correlations
            Cxx = C[:k, :k]
            Cyy = C[k0:k0+k, k0:k0+k]
            Cxy = C[:k, k0:k0+k]
            Cyx = C[k0:k0+k, :k]
    
            eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                            np.dot(np.linalg.pinv(Cyy), Cyx)))
    
            # Binary search if k is too large
            if not (np.all(np.isreal(eigs)) and
                    0 <= np.min(eigs) and
                    np.max(eigs) <= 1):
                ub -= 1
                k = (ub + lb) // 2
                continue
            if lb == ub: break
            lb = k
            if ub == lb + 1:
                k = ub
            else:
                k = (ub + lb) // 2
    
        return np.sqrt(np.max(eigs))
    
    
    