from numpy import*
from scipy.special import erf
import matplotlib.pyplot as plt

m_x = float(input("Give the value of m_x: "))
m_A = 0.02

def v_min(X): # X = E_R	
	mu = m_x*m_A/(m_x + m_A)
	return sqrt(X*m_A/(2*mu**2))
	
R0 = 7.186e-5/m_x
r = 4*m_A*m_x/(m_A + m_x)**2
E0 = 2.42e10*m_x
k_inv = 1.007282098 

def q(X):
	return sqrt(0.04*X)
	
def F_2(X):
	rn = 1.23
	return (3*exp(-((q(X)*0.9e-15)**2)/2)*(sin(q(X)*rn)-q(X)*rn*cos(q(X)*rn))/(q(X)*rn)**3)**2
	
def y(X):
	return (k_inv*R0*F_2(X)/(r*E0))*(0.42(erf((v_min(X) + 232e3)/220e3) - erf((v_min(X) - 232e3)/220e3))- 2.418e-3)

X = linspace(1e-3, 1, 500)
plt.plot(X, y(X))
plt.show()




