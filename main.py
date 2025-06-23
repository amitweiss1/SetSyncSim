import numpy as np
from typing import Set, Callable
from node import Node
# from minhash import MinHash
# from strata_estimator import  estimate_sym_diff_strata
from minhash_default import calc_symmetric_diff_minhash
from scipy.stats import norm, expon, gamma, uniform as scipy_uniform
from hyperloglog import calc_hyperloglog
import Distributaions as Distributions


class Distribution:
    def __init__(self, pdf: Callable[[float], float], expectation: float):
        self.pdf = pdf
        self.expectation = expectation


class ProbabilityDistributions:
    @staticmethod
    def normal(mean=0.5, std=0.1) -> Distribution:
        expectation = mean
        dist = norm(loc=mean, scale=std)
        norm_const = dist.cdf(1) - dist.cdf(0)
        pdf = lambda x: dist.pdf(x) / norm_const if 0 <= x <= 1 else 0
        return Distribution(pdf=pdf, expectation=expectation)

    @staticmethod
    def uniform(a=0, b=1) -> Distribution:
        expectation = (a + b) / 2
        dist = scipy_uniform(loc=a, scale=b - a)
        pdf = lambda x: dist.pdf(x) if a <= x <= b else 0
        return Distribution(pdf=pdf, expectation=expectation)

    @staticmethod
    def exponential(lambda_param=1) -> Distribution:
        expectation = 1 / lambda_param
        dist = expon(scale=1 / lambda_param)
        norm_const = dist.cdf(1) - dist.cdf(0)
        pdf = lambda x: dist.pdf(x) / norm_const if 0 <= x <= 1 else 0
        return Distribution(pdf=pdf, expectation=expectation)

    @staticmethod
    def gamma(alpha=2, beta=1) -> Distribution:
        expectation = alpha / beta
        dist = gamma(a=alpha, scale=1 / beta)
        norm_const = dist.cdf(1) - dist.cdf(0)
        pdf = lambda x: dist.pdf(x) / norm_const if 0 <= x <= 1 else 0
        return Distribution(pdf=pdf, expectation=expectation)


def sample_from_pdf(pdf: Callable[[float], float], max_pdf: float) -> float:
    while True:
        x = round(np.random.uniform(0, 1), 3)
        y = np.random.uniform(0, max_pdf)
        if y <= pdf(x):
            return x
# def sample_from_pdf_as_probability(pdf: Callable[[float], float], max_pdf: float) -> float:
#     x = round(np.random.uniform(0, 1), 3)
#     p = pdf(x) / max_pdf  # Normalize to [0,1]
#     return min(p, 1.0)

def generate_first_set(size: int, prob_func: Callable[[float], float], max_pdf:float) -> Set[Node]:
    result = set()
    for index in range(1, size + 1):
        new_node = Node(index)
        new_node.set_probability(sample_from_pdf(prob_func, max_pdf))
        result.add(new_node)
    return result


def generate_second_set(set1: Set[Node]) -> Set[Node]:
    set2 = set()
    for node in set1:
        new_node = Node(node.value)
        x = round(np.random.uniform(0, 1), 3)
        if x <= node.get_probability():
            set2.add(new_node)
    return set2


def calc_expectation(set1: Set[Node]) -> float:
    total = sum(node.get_probability() for node in set1)
    return total / len(set1) if set1 else 0.0


def calc_entropy(set1: Set[Node]) -> float:
    # Extract the probability values from the nodes
    probs = [node.get_probability() for node in set1]

    # Compute the total sum of probabilities
    total = sum(probs)

    # If all probabilities are zero, there is no information (entropy = 0)
    if total == 0:
        return 0.0

    # Normalize probabilities so they sum to 1 (required for entropy calculation)
    normalized_probs = [p / total for p in probs]

    # Compute entropy using the Shannon formula: H = -∑ p * log2(p)
    entropy = -sum(p * np.log2(p) for p in normalized_probs if p > 0)

    return entropy

def calculate_max_pdf(pdf: Callable[[float], float], resolution: int = 1000) -> float:
    x_values = np.linspace(0, 1, resolution)
    pdf_values = [pdf(x) for x in x_values]
    return max(pdf_values)


def run_simulation(set_size: int, prob_func: Callable[[float], float], max_pdf: float = 1.0,
                   num_permutations: int = 10, num_strata: int = 8,
                   distribution_name: str = "Unknown", expectation: float = 0.0):
    set1 = generate_first_set(set_size, prob_func, max_pdf)
    expectation = calc_expectation(set1)
    entropy = calc_entropy(set1)
    set2 = generate_second_set(set1)

    new_set1 = {node.value for node in set1}
    new_set2 = {node.value for node in set2}

    intersection = len(new_set1 & new_set2)
    union = len(new_set1 | new_set2)
    actual_diff = union - intersection

    # Estimate symmetric difference using SOTA algorithms estimator
    symmetric_diff_minhash = calc_symmetric_diff_minhash(new_set1, new_set2)
    symmetric_diff_hyperloglog = calc_hyperloglog(new_set1, new_set2)

    print(f"\n============== {distribution_name} Distribution ===============")
    print(f"Set 1 size: {len(new_set1)}")
    print(f"Set 2 size: {len(new_set2)}")
    print(f"Intersection size: {intersection}")
    print(f"Union size: {union}")
    print(f"\nActual diff size: {actual_diff:.4f}")
    print(f"Expectation of symmetric diff: {(1 - expectation) * len(new_set1):4f}")
    print(f"Entropy of Set 1: {entropy:.4f}")
    print(f"MinHash symmetric diff: {symmetric_diff_minhash:.4f}")
    print(f"HyperLogLog symmetric diff: {symmetric_diff_hyperloglog:.4f}")
    # print("=" * 50)


if __name__ == "__main__":
    print("Running simulation...")
    set_sizes = [10]

    mus = np.arange(0.1, 1, 0.1)

    for mu in mus:
        normal_params = Distributions.get_normal_params(mu)
        uniform_params = Distributions.get_uniform_params(mu)
        exponential_params = Distributions.find_lambda_for_exponential(mu)
        gamma_params = Distributions.find_gamma_params(mu)

        distributions = {
            "Normal": ProbabilityDistributions.normal(mean=normal_params["mean"], std=normal_params["std"]),
            "Uniform": ProbabilityDistributions.uniform(a=uniform_params["a"], b=uniform_params["b"]),
            "Exponential": ProbabilityDistributions.exponential(lambda_param=exponential_params["lambda"]),
            "Gamma": ProbabilityDistributions.gamma(alpha=gamma_params["alpha"], beta=gamma_params["lambda"])
        }

        for size in set_sizes:
            for dist_name, dist_config in distributions.items():
                prob_func = dist_config.pdf
                max_pdf = calculate_max_pdf(prob_func)
                print (f"\nExpectation {mu:.4f}")
                run_simulation(
                    set_size=size,
                    prob_func=prob_func,
                    max_pdf=max_pdf,
                    distribution_name=dist_name
                )


# Re-import required libraries after kernel reset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Simulated results storage
results = []

# Dummy simulation loop (replace with real simulation)
set_sizes = [10, 50, 100, 200, 500]
mus = np.arange(0.1, 1.0, 0.1)
distributions = ["Normal", "Uniform", "Exponential", "Gamma"]
algorithms = ["MinHash", "HyperLogLog"]

# Simulate some dummy data for demonstration
np.random.seed(42)
for dist in distributions:
    for mu in mus:
        for size in set_sizes:
            for _ in range(5):  # multiple runs per setting
                entropy = np.random.uniform(0.5, 3.5)
                actual_diff = size * (1 - mu)
                for algo in algorithms:
                    est_diff = actual_diff * np.random.normal(1.0, 0.1)  # add some noise
                    error_percent = abs(est_diff - actual_diff) / actual_diff * 100 if actual_diff > 0 else 0
                    results.append({
                        "Distribution": dist,
                        "Mean": mu,
                        "SetSize": size,
                        "Entropy": entropy,
                        "Algorithm": algo,
                        "ErrorPercent": error_percent
                    })

def print_plots():
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Plot 1: Error vs. Set Size
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="SetSize", y="ErrorPercent", hue="Algorithm", style="Algorithm", markers=True)
    plt.title("Prediction Error vs. Set Size")
    plt.ylabel("Error [%]")
    plt.xlabel("Set Size")
    plt.grid(True)
    plt.show()

    # Plot 2: Error vs. Entropy (scatter)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Entropy", y="ErrorPercent", hue="Algorithm", alpha=0.7)
    sns.lineplot(data=df, x="Entropy", y="ErrorPercent", hue="Algorithm", estimator='mean', lw=2, legend=False)
    plt.title("Prediction Error vs. Entropy")
    plt.ylabel("Error [%]")
    plt.xlabel("Entropy")
    plt.grid(True)
    plt.show()

    # Plot 3: Error vs. Mean
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Mean", y="ErrorPercent", hue="Algorithm", style="Algorithm", markers=True)
    plt.title("Prediction Error vs. Expectation (μ)")
    plt.ylabel("Error [%]")
    plt.xlabel("Expectation (μ)")
    plt.grid(True)
    plt.show()

    # Plot 4: Separate per distribution
    for dist in distributions:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df[df["Distribution"] == dist], x="SetSize", y="ErrorPercent", hue="Algorithm", style="Algorithm", markers=True)
        plt.title(f"Prediction Error vs. Set Size for {dist} Distribution")
        plt.ylabel("Error [%]")
        plt.xlabel("Set Size")
        plt.grid(True)
        plt.show()
