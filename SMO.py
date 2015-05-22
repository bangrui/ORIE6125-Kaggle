#################################################################################
#                                                                               #
#   Sequential Minimal Optimization for SVM                                     #
#                                                                               # 
#   Author: Pu Yang, Bangrui Chen                                               #
#   V1.0                                                                        #
#                                                                               #
#   2015/05/20                                                                  #
#################################################################################

"""
v1.0: (1) Solves the dual QP of a standard (linear) soft margin SVM
        (2) User specified regularizing parameter C and a tolerence "tol" of the violation 
        of KKT conditions, and a parameter "eps" controls the negligibility of updates
        (3) General kernel is not implemented, but can be easily included 
"""

import numpy as np

class SMO:

    def __init__(self, X, Y, C, tol, eps):
        self._X = X
        self._Y = Y
        self._C = C
        self._tol = tol
        self._eps = eps
        self._nsamples, self._nfeatures = self._X.shape
        self._deltab = 0
        self._alpha = np.zeros(self._nsamples)
        self._weight = np.zeros(self._nfeatures)
        self._b = 0
        self._errorCache = -self._Y #stores w^Tx + b - y for all (x,y) in (X,Y)
    
    def get_X(self):
        return self._X

    def get_Y(self):
        return self._Y

    def get_alpha(self):
        return self._alpha

    def get_weight(self):
        return self._weight

    def get_threshold(self):
        return self._b

    def get_tol(self):
        return self._tol
    
    def get_eps(self):
        return self._eps

    def get_C(self):
        return self._C

    def get_errorCache(self):
        return self._errorCache
    
    def set_X(self, X):
        self._X = X

    def set_Y(self, Y):
        self._Y = Y

    def set_C(self, C):
        self._C = C
     
    def set_tol(self, tol):
        self._tol = tol
    
    def set_eps(self, eps):
        self._eps = eps


    """
    The main routine of SMO, solves the dual QP and updates alpha, w, b. 
    """
    def SMO_main(self):
        numChanged = 0
        examineAll = True

        while numChanged > 0 or examineAll:
            numChanged = 0
            if examineAll:
                for k in np.arange(self._nsamples):
                    numChanged += self.__examine(k)
            else:
                for k in np.arange(self._nsamples):
                    if self._alpha[k] != 0 and self._alpha[k] != self._C:
                        numChanged += self.__examine(k)
            if examineAll:
                examineAll = False            
            elif numChanged == 0:
                examineAll = True
        return



    """
    Given the first alpha being alpha_k, pick the second alpha and optimize them by calling __optAlpha().
    """
    def __examine(self, i1):
        y1 = self._Y[i1];
        alpha_1 = self._alpha[i1]

        if alpha_1 > 0 and alpha_1 < self._C:
            E1 = self._errorCache[i1]
        else:
            E1 = self.fun_eval(i1) - y1
        
        r1 = y1*E1
        
        #Check KKT conditions
        if (r1 < -self._tol and alpha_1 < self._C) or (r1 > self._tol and alpha_1 > 0):
            
            #Choose alpha_2 from non-bounded examples, so that E1-E2 is maximized
            max_gapE = 0
            i2 = -1
            for k in np.arange(self._nsamples):
                if self._alpha[k] >0 and self._alpha[k] < self._C:
                    E2 = self._errorCache[k]
                    temp = abs(E2 - E1)
                    if temp > max_gapE:
                        max_gapE = temp
                        i2 = k
            if i2 >= 0:
                if self.__optAlpha(i1, i2):
                    return 1
            
            #If no improvement, try all non bound examples
            for k in np.arange(self._nsamples):
                i2 = k
                if self._alpha[i2] > 0 and self._alpha[i2] < self._C:
                    if self.__optAlpha(i1, i2):
                        return 1
             
            #If no improvement, try all examples
            for k in np.arange(self._nsamples):
                i2 = k
                if self.__optAlpha(i1, i2):
                    return 1
        return 0
        

    """
    Given the index i1, i2, optimize with respect to alpha_{i1}, alpha_{i2}.  
    """
    def __optAlpha(self, i1, i2):
        if i1 == i2:
            return False
        alpha_1 = self._alpha[i1]
        alpha_2 = self._alpha[i2]
        y1 = self._Y[i1]
        y2 = self._Y[i2]
        E1 = 0
        E2 = 0
        if alpha_1 > 0 and alpha_1 < self._C:
            E1 = self._errorCache[i1]
        else: 
            E1 = self.fun_eval(i1) - y1
        if alpha_2 >0 and alpha_2 < self._C:
            E2 = self._errorCache[i2]
        else:
            E2 = self.fun_eval(i2) - y2
        a1 = 0
        a2 = 0
        s = y1 * y2
        
        #Compute upper and lower bounds for alpha_{i2}
        L, H = self.__computeLH(y1, y2, alpha_1, alpha_2)

        if L == H:
            return False
        
        #Update alpha_{i1}, alpha_{i2}
        eta, K11, K12, K22 = self.__computeEta(i1, i2)
        if eta < 0:
            a2 = alpha_2 + y2* (E2 - E1)/eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            #Evaluate the function values at the bounds and pick the one with larger value
            Lobj, Hobj = self.__comuteLHObj(y1, y2, eta, alpha_2, E1, E2, L, H)
            if Lobj > Hobj + self._eps:
                a2 = L
            elif Lobj < Hobj - self._eps:
                a2 = H
            else:
                a2 = alpha_2
        
        #Check if the change is negligible
        if abs(a2 - alpha_2) < self._eps*(a2 + alpha_2 + self._eps):
            return False

        a1 = alpha_1 - s*(a2 - alpha_2)
        if a1 < 0:
            a2 += s*a1
            a1 = 0
        elif a1 > self._C:
            a2 += s*(a1 - self._C)
            a1 = self._C

        delta_b = self.__UpdateThreshold(y1, y2, a1, a2, E1, E2, K11, K12, K22, alpha_1, alpha_2)
        self.__UpdateWeight(i1, i2, y1, y2, a1, a2, alpha_1, alpha_2)
        self.__UpdateErrorCache(i1, i2, y1, y2, a1, a2, alpha_1, alpha_2, delta_b)

        self._alpha[i1] = a1
        self._alpha[i2] = a2
        return True


    def __computeLH(self, y1, y2, alpha_1, alpha_2):
        if(y1 == y2):
            gamma = alpha_1 + alpha_2
            if gamma > self._C:
                L = gamma - self._C
                H = self._C
            else:
                L = 0
                H = gamma
        else:
            gamma = alpha_1 - alpha_2
            if gamma > 0:
                L = 0
                H = self._C - gamma
            else:
                L = -gamma
                H = self._C
        return [L, H]


    def __computeEta(self, i1, i2):
        x1 = self._X[i1]
        x2 = self._X[i2]
        k11 = np.inner(x1, x1)
        k12 = np.inner(x1, x2)
        k22 = np.inner(x2, x2)
        eta = 2*k12 - k11 - k22
        return eta, k11, k12, k22


    def __computeLHObj(self, y1, y2, eta, alpha_2, E1, E2, L, H):
        c1 = eta/2
        c2 = y2 * (E1 - E2) - eta * alpha_2
        Lobj = c1 * L^2 + c2 * L
        Hobj = c1 * H^2 + c2 * H
        return Lobj, Hobj


    def __UpdateThreshold(self, y1, y2, a1, a2, E1, E2, K11, K12, K22, alpha_1, alpha_2):
        b = self._b
        bnew = b
        if a1 > 0 and a1 < self._C:
            bnew = b + E1 + y1 * (a1 - alpha_1) * K11 + y2 * (a2 - alpha_2) * K12
        else:
            if a2 > 0 and a2 < self._C: 
                bnew = b + E2 + y1 * (a1 - alpha_1) * K12 + y2 * (a2 - alpha_2) * K22
            else:
                b1 = b + E1 + y1 * (a1 - alpha_1) * K11 + y2 * (a2 - alpha_2) * K12
                b2 = b + E2 + y1 * (a1 - alpha_1) * K12 + y2 * (a2 - alpha_2) * K22
                bnew = (b1 + b2)/2
        delta_b = bnew - b
        self._b = bnew
        return delta_b

    
    def __UpdateWeight(self, i1, i2, y1, y2, a1, a2, alpha_1, alpha_2):
        t1 = y1 * (a1 - alpha_1) 
        t2 = y2 * (a2 - alpha_2)
        delta_w = self._X[i1] * t1 + self._X[i2] * t2
        self._weight += delta_w


    def __UpdateErrorCache(self, i1, i2, y1, y2, a1, a2, alpha_1, alpha_2, delta_b):
        t1 = y1 * (a1 - alpha_1)
        t2 = y2 * (a2 - alpha_2)
        for k in np.arange(self._nsamples):
            if 0 < self._alpha[k] and self._alpha[k] < self._C:
                self._errorCache[k] += t1 * np.inner(self._X[i1], self._X[k]) + t2 * np.inner(self._X[i2], self._X[k]) - delta_b
        self._errorCache[i1] = 0
        self._errorCache[i2] = 0
        
    
    """
    Given an index i1, evaluates w^T* x_{i1} + b with currently learned w and b
    """
    def fun_eval(self, i1):
        return np.inner(self._weight, self._X[i1]) - self._b
    




