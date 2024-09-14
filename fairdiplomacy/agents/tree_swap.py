import numpy as np
import matplotlib.pyplot as plt
import os


# ----- Regret Computation Functions -----

def compute_alg_cumulative_loss(losses, distributions):
    T = len(losses)
    N = len(losses[0])

    cumul_losses = []
    L_A = 0
    for t in range(T):
        L_A += np.dot(losses[t], distributions[t])
        cumul_losses.append(L_A)

    return cumul_losses

def compute_external_regret(losses, distributions):
    T = len(losses)
    N = len(losses[0])
    cum_regret = np.zeros(T)

    L_A = 0
    L_strategies = np.zeros(N)

    for t in range(T):
        L_A += np.dot(losses[t], distributions[t])
        L_strategies += losses[t]
        L_min = float('inf')
        for i in range(N):
            L_min = min(L_min, L_strategies[i])
        cum_regret[t] = (L_A - L_min)/(t+1)
    return cum_regret

def compute_swap_regret(losses, distributions):
    T = len(losses)
    N = len(losses[0])

    alg_cumul_losses = compute_alg_cumulative_loss(losses, distributions)
    
    loss_sum = [0.0 for _ in range(T)]
    for i in range(N):
        min_i = [float('inf') for _ in range(T)]
        for j in range(N):
            ij = 0.0
            for t in range(T):
                ij += distributions[t][i] * losses[t][j]
                min_i[t] = min(min_i[t], ij)
        for t in range(T):
            loss_sum[t] += min_i[t]
        print("swap regret i:", i)

    swap_regrets_per_time_step = []
    for t in range(T):
        L_A = alg_cumul_losses[t]
        L_swap_min = loss_sum[t]
        total_swap_regret = L_A - L_swap_min
        swap_regrets_per_time_step.append(total_swap_regret/(t+1))

    return swap_regrets_per_time_step

# ----- RWM Helper Functions -----

def RWM_weights(prev_weights, prev_loss, eta):
    weights = np.full(len(prev_weights), 1.0)
    for i in range(len(prev_weights)):
        weights[i] = prev_weights[i] * (1 - eta) ** prev_loss[i]
    if min(weights) < 1e-13 and max(weights) < 1e100:
        weights *= 10
    return weights

def RWM_distribution(weights):
    total_weight = np.sum(weights)
    distribution = weights / total_weight
    return distribution

# ----- BM Helper Functions -----

def stationary_distribution(matrix, tol=1e-10, max_iter=10000):
    n = matrix.shape[0]
    distribution = np.ones(n) / n
    if np.sum(matrix)/n > 1.001 or np.sum(matrix)/n < 0.999:
        raise ValueError("Stationary distribution did not converge")

    for t in range(max_iter):
        new_distribution = distribution @ matrix
        if sum(new_distribution) > 1.001 or sum(new_distribution) < 0.999:
            raise ValueError("Stationary distribution did not converge")
        if np.linalg.norm(new_distribution - distribution) < tol:
            # print("Stationary distribution converged after", t, "iterations")
            return new_distribution
        distribution = new_distribution

    # print(matrix)
    raise ValueError("Stationary distribution did not converge")

# ----- TreeSwap Helper Functions -----

def get_base_M_representation(x, M, d):
    bits = []
    while x > 0:
        bits.append(x % M)
        x //= M
    bits += [0] * (d - len(bits))
    vals = [1 if bits[0] > 0 else 0]
    for i in range(1, d):
        if vals[i - 1] != 0 or bits[i] > 0:
            vals.append(1)
        else:
            vals.append(0)
    return bits[::-1], vals[::-1]

# ----- General Helper Functions -----

def observe_1p(x, t, losses):
    f = losses[t]
    return f

def observe_2p(strategy1, strategy2, A, t=None):
    loss1 = np.dot(A, strategy1)
    loss2 = np.dot(A.T, strategy2)
    return loss1, loss2

def truncate_regret(regret, n=10):
    return [np.nan if i < n else r for i, r in enumerate(regret)]

# ----- Helper Class -----

class RWM():
    def __init__(self, eta, id=-1, N=100):
        self._N = N
        self._cur_action = np.ones(self._N) / self._N
        self._weights = np.ones(self._N)
        self._id = id
        self._eta = eta
    
    def act(self, t):
        self._cur_action = RWM_distribution(self._weights)
        return self._cur_action
    
    def update(self, f):
        self._weights = RWM_weights(self._weights, f, self._eta)

class ORWM(RWM): # optimistic RWM
    def __init__(self, eta, id=-1, N=100):
        super().__init__(eta, id, N)
        self._m = np.zeros(self._N)

    def update(self, loss):
        w = 2 * loss - self._m
        self._weights = RWM_weights(self._weights, w, self._eta)
        self._m = loss

# ----- No-Regret Algorithms -----

class RWMAlg(RWM):
    def __init__(self, eta, id=-1, N=100):
        super().__init__(eta, id, N)
        self._x = []
        self._f = []
        self._name = "RWM"
    
    def act(self, t):
        if t % 1000 == 0:
            print("RWM iteration", t)
        self._x.append(RWM_distribution(self._weights))
        return self._x[-1]

    def update(self, loss):
        super().update(loss)
        self._f.append(loss)

class ORWMAlg(ORWM):
    def __init__(self, eta, id=-1, N=100):
        super().__init__(eta, id, N)
        self._x = []
        self._f = []
        self._name = "ORWM"
    
    def act(self, t):
        if t % 1000 == 0:
            print("ORWM iteration", t)
        self._x.append(RWM_distribution(self._weights))
        return self._x[-1]

    def update(self, loss):
        super().update(loss)
        self._f.append(loss)

class BMAlg():
    def __init__(self, ext_alg, eta, N):
        self._N = N
        self._x = []
        self._f = []
        self._algs = [ext_alg(eta, id=i, N=N) for i in range(self._N)]
        self._name = f"BM-{ext_alg.__name__}"

    def act(self, t):
        if t % 1000 == 0:
            print("BM iteration", t, self._N)
        Q = np.full((self._N, self._N), 1/self._N)
        for i in range(self._N):
            Q[i] = self._algs[i].act(t)
        p = stationary_distribution(Q)
        self._x.append(p)
        return self._x[-1]
    
    def update(self, loss):
        for i in range(self._N):
            p = self._x[-1]
            loss_i = p[i] * loss
            self._algs[i].update(loss_i)
        self._f.append(loss)

class TreeSwap():
    def __init__(self, ext_alg, eta, N, M, d):
        self._M = M
        self._d = d
        self._x = []
        self._f = []
        self._algs = [ext_alg(eta, id=i, N=N) for i in range(self._d)]
        self._name = f"TreeSwap-{ext_alg.__name__}"
    
    def tree_swap(self, t):
        algs = self._algs
        bits, vals = get_base_M_representation(t, self._M, self._d)
        for h in range(1, self._d+1):
            if h == self._d or vals[h] == 0:
                num_rounds = self._M ** (self._d - h)
                if t >= num_rounds:
                    avg_f = sum(self._f[t - num_rounds : t]) / num_rounds
                    algs[h-1].update(avg_f)
                algs[h-1].act(t)
        x = sum([algs[h]._cur_action for h in range(self._d)]) / self._d
        self._x.append(x)

    def act(self, t):
        if t % 1000 == 0:
            print("TreeSwap iteration", t)
        self.tree_swap(t)
        return self._x[-1]
    
    def update(self, loss):
        self._f.append(loss)

# ----- Debugging -----

def print_alg(alg):
    print(alg._name, "x:")
    # print(alg._x[:10])
    # print("----------------")
    print(alg._x[-10:])

# ----- Game Simulation -----

def simulate_1p(g, alg):
    for t in range(g.T):
        x = alg.act(t)
        f = observe_1p(x, t, g.losses)
        alg.update(f)

def simulate_2p(g, A, alg1, alg2):
    for t in range(g.T):
        x1 = alg1.act(t)
        x2 = alg2.act(t)
        loss1, loss2 = observe_2p(x1, x2, A)
        alg1.update(loss1)
        alg2.update(loss2)

# ----- Main -----

def run_1p(g):
    alg1 = RWMAlg(g.eta1, N=g.N)
    alg2 = BMAlg(RWM, g.eta1, N=g.N)
    alg3 = TreeSwap(RWM, g.eta2, g.N, g.M, g.d)
    alg4 = ORWMAlg(g.eta1, N=g.N)
    alg5 = BMAlg(ORWM, g.eta1, N=g.N)
    alg6 = TreeSwap(ORWM, g.eta2, g.N, g.M, g.d)
    simulate_1p(g, alg1)
    simulate_1p(g, alg2)
    simulate_1p(g, alg3)
    simulate_1p(g, alg4)
    simulate_1p(g, alg5)
    simulate_1p(g, alg6)

    ext_regret1 = truncate_regret(compute_external_regret(alg1._f, alg1._x))
    ext_regret2 = truncate_regret(compute_external_regret(alg2._f, alg2._x))
    ext_regret3 = truncate_regret(compute_external_regret(alg3._f, alg3._x))
    ext_regret4 = truncate_regret(compute_external_regret(alg4._f, alg4._x))
    ext_regret5 = truncate_regret(compute_external_regret(alg5._f, alg5._x))
    ext_regret6 = truncate_regret(compute_external_regret(alg6._f, alg6._x))

    swap_regret1 = truncate_regret(compute_swap_regret(alg1._f, alg1._x))
    swap_regret2 = truncate_regret(compute_swap_regret(alg2._f, alg2._x))
    swap_regret3 = truncate_regret(compute_swap_regret(alg3._f, alg3._x))
    swap_regret4 = truncate_regret(compute_swap_regret(alg4._f, alg4._x))
    swap_regret5 = truncate_regret(compute_swap_regret(alg5._f, alg5._x))
    swap_regret6 = truncate_regret(compute_swap_regret(alg6._f, alg6._x))

    # print_alg(alg1)
    # print_alg(alg2)
    # print_alg(alg3)

    plt.figure(figsize=(12, 5))
    plt.plot(ext_regret1, label=f"{alg1._name} External Regret")
    plt.plot(swap_regret1, label=f"{alg1._name} Swap Regret")
    plt.plot(ext_regret2, label=f"{alg2._name} External Regret")
    plt.plot(swap_regret2, label=f"{alg2._name} Swap Regret")
    plt.plot(ext_regret3, label=f"{alg3._name} External Regret")
    plt.plot(swap_regret3, label=f"{alg3._name} Swap Regret")
    plt.plot(ext_regret4, label=f"{alg4._name} External Regret")
    plt.plot(swap_regret4, label=f"{alg4._name} Swap Regret")
    plt.plot(ext_regret5, label=f"{alg5._name} External Regret")
    plt.plot(swap_regret5, label=f"{alg5._name} Swap Regret")
    plt.plot(ext_regret6, label=f"{alg6._name} External Regret")
    plt.plot(swap_regret6, label=f"{alg6._name} Swap Regret")
    plt.xlabel("Time")
    plt.ylabel("Regrets")
    plt.title("External and Swap Regret Over Time")
    plt.legend()
    plt.show()

def run_2p(g, alg1, alg2):
    data = np.load('random100.npz')
    A = data['A']

    simulate_2p(g, A, alg1, alg2)

    ext_regret1 = truncate_regret(compute_external_regret(alg1._f, alg1._x))
    ext_regret2 = truncate_regret(compute_external_regret(alg2._f, alg2._x))
    swap_regret1 = truncate_regret(compute_swap_regret(alg1._f, alg1._x))
    swap_regret2 = truncate_regret(compute_swap_regret(alg2._f, alg2._x))

    plt.figure(figsize=(12, 5))
    plt.plot(ext_regret1, label=f"{alg1._name} External Regret")
    plt.plot(swap_regret1, label=f"{alg1._name} Swap Regret")
    plt.plot(ext_regret2, label=f"{alg2._name} External Regret")
    plt.plot(swap_regret2, label=f"{alg2._name} Swap Regret")
    plt.xlabel("Time")
    plt.ylabel("Regrets")
    plt.title("External and Swap Regret Over Time")
    plt.legend()
    dir_name = "plots/N=" + str(g.N) + "M=" + str(g.M) + "d=" + str(g.d) + "/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(dir_name + alg1._name + "_" + alg2._name + ".png")

if __name__ == "__main__":
    class Globals:
    ### class to hold all "global" variables  ###
        def __init__(self) -> None:
            self.T = 4000
            self.N = 100

            self.M = np.round(np.log(self.T)).astype(int)
            self.d = np.round(np.log(self.T) / np.log(self.M)).astype(int) + 1

            self.eta1 = np.sqrt(np.log(self.N) / self.T) # for RWM, BM
            self.eta2 = np.sqrt(np.log(self.N) / self.M) # for TreeSwap

            self.losses = np.random.rand(self.T+1, self.N) # for 1p
    g = Globals()
    # run_1p(g)

    run_2p(g, RWMAlg(g.eta1, N=g.N), BMAlg(RWM, g.eta1, N=g.N))
    run_2p(g, TreeSwap(ORWM, g.eta2, g.N, g.M, g.d), BMAlg(RWM, g.eta1, N=g.N))
    run_2p(g, RWMAlg(g.eta1, N=g.N), TreeSwap(ORWM, g.eta2, g.N, g.M, g.d))
    run_2p(g, RWMAlg(g.eta1, N=g.N), ORWMAlg(g.eta1, N=g.N))
    run_2p(g, BMAlg(RWM, g.eta1, N=g.N), BMAlg(RWM, g.eta1, N=g.N))
    run_2p(g, TreeSwap(ORWM, g.eta2, g.N, g.M, g.d), TreeSwap(ORWM, g.eta2, g.N, g.M, g.d))