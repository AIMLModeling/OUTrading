import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy import sparse
from scipy.sparse.linalg import spsolve
np.random.seed(seed=42)
# define Ornstein-Uhlenbeck Process (OU Process)
class OU_process:
    """
    Class for the OU process:
    theta = long term mean
    sigma = diffusion coefficient
    kappa = mean reversion coefficient
    """
    def __init__(self, sigma=0.2, theta=-0.1, kappa=0.1):
        self.theta = theta
        if sigma < 0 or kappa < 0:
            raise ValueError("sigma,theta,kappa must be positive")
        else:
            self.sigma = sigma
            self.kappa = kappa
    def path(self, X0=0, T=1, N=10000, paths=1):
        """
        Produces a matrix of OU process:  X[N, paths]
        X0 = starting point
        N = number of time points (there are N-1 time steps)
        T = Time in years
        paths = number of paths
        """
        dt = T / (N - 1)
        X = np.zeros((N, paths))
        X[0, :] = X0
        W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

        std_dt = np.sqrt(self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * dt)))
        for t in range(0, N - 1):
            X[t + 1, :] = self.theta + np.exp(-self.kappa * dt) * (X[t, :] - self.theta) + std_dt * W[t, :]

        return X
# Parameters and Time Vector Initialization
N = 1000  # time steps
paths = 5000  # simulated paths
X0 = 0  # initial position
kappa = 10
theta = 0
sigma = 2
T = 1  # terminal time
std_asy = np.sqrt(sigma**2 / (2 * kappa))  # open position
std_10 = std_asy / 10  # close position
np.random.seed(seed=66)
OU = OU_process(sigma=sigma, theta=theta, kappa=kappa)  # creates the OU object
X = OU.path(X0=X0, T=T, N=N, paths=paths)  # path simulation
process = 0  # the process for the plot
T_vec, dt = np.linspace(0, T, N, retstep=True)
# Implements a simple trading strategy based on the OU process.
# The strategy goes long or short on the asset depending on whether the process
# exceeds certain thresholds (`std_open` for opening positions and `std_close` for
# closing positions). 
def strategy(X, mean=0, std_open=0.5, std_close=0.05, TC=0):
    """Implementation of the strategy.
        - std_open = levels for opening the position.
        - std_close = levels for closing the position.
        - TC = Transaction costs
    Returns:
        - status:  at each time says if we are long=1, short=-1 or we have no open positions = 0
        - cash: the cumulative amount of cash gained by the strategy.
        At terminal time if there is an open position, it is closed.
    """
    status = np.zeros_like(X)
    cash = np.zeros_like(X)
    cash[0] = X0
    for i, x in enumerate(X):
        if i == 0:
            continue
        if (status[i - 1] == 1) and (x >= mean - std_close):
            status[i] = 0
            cash[i] += x * (1 + TC)
        elif (status[i - 1] == -1) and (x <= mean + std_close):
            status[i] = 0
            cash[i] -= x * (1 + TC)
        elif (status[i - 1] == 0) and (x >= mean + std_open):
            status[i] = -1
            cash[i] += x * (1 + TC)
        elif (status[i - 1] == 0) and (x <= mean - std_open):
            status[i] = 1
            cash[i] -= x * (1 + TC)
        else:
            status[i] = status[i - 1]
    if status[-1] == 1:
        cash[-1] += x * (1 + TC)
    if status[-1] == -1:
        cash[-1] -= x * (1 + TC)

    return status, cash.cumsum()
status, cash = strategy(X[:, process], mean=theta, std_open=std_asy, std_close=std_10, TC=0)
PnL = []  # Profit and loss for this strategy
for i in range(paths):
    PnL.append(strategy(X[:, i], mean=theta, std_open=std_asy, std_close=std_10, TC=0)[1][-1])
PnL = np.array(PnL)
fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(222)
ax4 = fig.add_subplot(224)
ax1.plot(T_vec, X[:, process], linewidth=0.5)
ax1.plot(T_vec, (theta + std_asy) * np.ones_like(T_vec), label="open short position", color="sandybrown")
ax1.plot(T_vec, (theta - std_asy) * np.ones_like(T_vec), label="open long position", color="chocolate")
ax1.plot(T_vec, (theta + std_10) * np.ones_like(T_vec), label="close short position", color="gray")
ax1.plot(T_vec, (theta - std_10) * np.ones_like(T_vec), label="close long position", color="rosybrown")
ax1.plot(T_vec, theta * np.ones_like(T_vec), label="Long term mean", color="black")
ax1.legend()
ax1.set_title(f"OU process: process {process}")
ax1.set_xlabel("T")
ax2.plot(T_vec, status, linestyle="dashed", color="grey")
ax2.set_title("Strategy: 1=LONG, -1=SHORT, 0=No open positions ")
ax3.hist(PnL, density=True, bins=100, facecolor="LightBlue", label="frequencies")
x = np.linspace(PnL.min(), PnL.max(), 100)
ax3.plot(x, ss.norm.pdf(x, loc=PnL.mean(), scale=PnL.std()), color="r", label="Normal density")
SR = PnL.mean() / PnL.std()
ax3.legend()
ax3.set_title(f"PnL distribution. Sharpe ratio SR = {SR.round(2)}")
ax4.plot(T_vec, cash)
ax4.set_title("Cumulative amount of cash in the portfolio")
plt.show()

T_to_asy = np.argmax(np.logical_or(X <= -std_asy, X >= std_asy), axis=0) * dt  # first exit time
print("Are there paths that never exit the strip?", (T_to_asy == 0).any())  # argmax returns 0 if it can't find
print(f"The expected time is {T_to_asy.mean()} with standard error {ss.sem(T_to_asy)}")
Nspace = 100000  # space steps
x_max = theta + std_asy
x_min = theta - std_asy
x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)  # space discretization
U = np.zeros(Nspace)  # grid initialization
constant_term = -np.ones(Nspace - 2)  # -1
# construction of the tri-diagonal matrix D
sig2 = sigma * sigma
dxx = dx * dx
max_part = np.maximum(kappa * (theta - x), 0)  # upwind positive part
min_part = np.minimum(kappa * (theta - x), 0)  # upwind negative part
a = -min_part / dx + 0.5 * sig2 / dxx
b = (min_part - max_part) / dx - sig2 / dxx
c = max_part / dx + 0.5 * sig2 / dxx
aa = a[1:]
cc = c[:-1]  # upper and lower diagonals
D = sparse.diags([aa, b, cc], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()  # matrix D
U[1:-1] = spsolve(D, constant_term)
fig = plt.figure(figsize=(12, 5))
plt.plot(x, U, label="Expected time curve")
plt.plot([X0, X0], [0, 0.06], "k--", label="starting point in the simulation")
plt.plot([x_max, x_max], [0, 0.06], "g--", label="$B_2$")
plt.plot([x_min, x_min], [0, 0.06], "g--", label="$B_1$")
plt.legend(loc="upper right")
plt.xlabel("Starting point, X0")
plt.title("Expected time to exit the strip")
plt.show()
expected_t_ode = np.interp(X0, x, U)
print(f"The expected time starting at X_0={X0},computed by ODE is {expected_t_ode}")
Nspace = 6000  # M space steps
Ntime = 8000  # N time steps
x_max = theta + std_asy  # B2
x_min = theta - std_asy  # B1
x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)  # space discretization
T_array, Dt = np.linspace(0, T, Ntime, retstep=True)  # time discretization
Payoff = 0  # payoff

V = np.zeros((Nspace, Ntime))  # grid initialization
offset = np.zeros(Nspace - 2)  # vector to be used for the boundary terms
V[:, 0] = Payoff  # initial condition
V[-1, :] = 1  # lateral boundary condition
V[0, :] = 1  # lateral boundary condition
# construction of the tri-diagonal matrix D
sig2 = sigma * sigma
dxx = dx * dx
max_part = np.maximum(kappa * (theta - x[1:-1]), 0)  # upwind positive part
min_part = np.minimum(kappa * (theta - x[1:-1]), 0)  # upwind negative part
a = min_part * (Dt / dx) - 0.5 * (Dt / dxx) * sig2
b = 1 + (Dt / dxx) * sig2 + Dt / dx * (max_part - min_part)
c = -max_part * (Dt / dx) - 0.5 * (Dt / dxx) * sig2
a0 = a[0]
cM = c[-1]  # boundary terms
aa = a[1:]
cc = c[:-1]  # upper and lower diagonals
D = sparse.diags([aa, b, cc], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()  # matrix D
for n in range(Ntime - 1):
    # forward computation
    offset[0] = a0 * V[0, n]
    offset[-1] = cM * V[-1, n]
    V[1:-1, n + 1] = spsolve(D, (V[1:-1, n] - offset))
fn = RegularGridInterpolator((x, T_array), V, bounds_error=False, fill_value=None)  # interpolator at X0
Cumulative = fn((X0, T_array))  # Cumulative at x=X0
distribution = (Cumulative[1:] - Cumulative[:-1]) / Dt  # Density
fig = plt.figure(figsize=(12, 5))
plt.xlabel("Time")
plt.title("Distribution of the first exit time from the strip")
plt.hist(T_to_asy, density=True, bins=100, facecolor="LightBlue", label="histogram from the MC simulation")
plt.plot(T_array[:-1], distribution, label="Density from the PDE method")
plt.axvline(expected_t_ode, label="expected time")
plt.xlim([0, 0.4])
plt.legend()
plt.show()
exp_t_integral = distribution @ T_array[:-1] * Dt  # integral of t*density(t)*dt
print(f"Expected value from density tau={exp_t_integral} corresponds to PDE tau={expected_t_ode}")
