import numpy as np
import fractions

class Markov:
    def __init__(self, lam, P):
        self.lam = np.array(lam, dtype=float)  #initial probabilities
        self.P = np.array(P, dtype=float)      #transition matrix
    def prob(self, n, j):
        """ Pr(X_n = j) = (lambda P^n)_j """
        prob = (self.lam@(np.linalg.matrix_power(self.P,n)))[j-1]
        prob_frac = fractions.Fraction(prob).limit_denominator()
        num, denom = prob_frac.numerator, prob_frac.denominator
        print(f"P(X_{n} = {j}) = {num}/{denom}")
        return prob
    def prob_i(self, i, n, j):
        """ Pr_i(X_n = j) = Pr(X_{n+m} = j | X_m = i) = (P^n)_{ij}"""
        prob = (np.linalg.matrix_power(self.P,n))[i-1][j-1]
        prob_frac = fractions.Fraction(prob).limit_denominator()
        num, denom = prob_frac.numerator, prob_frac.denominator
        print(f"P_{i}(X_{n} = {j}) = {num}/{denom}")
        return prob