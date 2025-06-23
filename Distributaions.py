import numpy as np
from scipy.stats import norm, uniform, expon, gamma
from scipy.optimize import minimize_scalar

# ===== Utility: truncated expectation on [0, 1] =====
def truncated_expectation(dist, a=0, b=1, num_points=1000):
    norm_const = dist.cdf(b) - dist.cdf(a)
    if norm_const == 0:
        return 0
    xs = np.linspace(a, b, num_points)
    pdf_vals = dist.pdf(xs) / norm_const
    return np.trapz(xs * pdf_vals, xs)

# ===== 1. Normal distribution: E[X] = mean =====
def get_normal_params(target_mu, fixed_std=0.1):
    return {"mean": target_mu, "std": fixed_std}

# ===== 2. Uniform distribution: E[X] = (a + b) / 2 =====
def get_uniform_params(target_mu):
    if target_mu <= 0.5:
        a, b = 0.0, 2.0 * target_mu
    else:
        b, a = 1.0, 2.0 * target_mu - 1
    return {"a": a, "b": b}

# ===== 3. Exponential (truncated) =====
def find_lambda_for_exponential(target_mu):
    def objective(lamb):
        if lamb <= 0:
            return np.inf
        dist = expon(scale=1 / lamb)
        exp = truncated_expectation(dist)
        return abs(exp - target_mu)
    res = minimize_scalar(objective, bounds=(1e-3, 100), method='bounded')
    return {"lambda": res.x}

# ===== 4. Gamma (with alpha = mu * lambda) =====
def find_gamma_params(target_mu):
    def objective(lamb):
        if lamb <= 0:
            return np.inf
        alpha = target_mu * lamb
        dist = gamma(a=alpha, scale=1 / lamb)
        exp = truncated_expectation(dist)
        return abs(exp - target_mu)
    res = minimize_scalar(objective, bounds=(1e-3, 100), method='bounded')
    lambda_opt = res.x
    alpha_opt = target_mu * lambda_opt
    return {"alpha": alpha_opt, "lambda": lambda_opt}

