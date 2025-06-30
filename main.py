import numpy as np
from typing import Set, Callable
from node import Node
# from minhash import MinHash
# from strata_estimator import  estimate_sym_diff_strata
from minhash_default import calc_symmetric_diff_minhash
from scipy.stats import norm, expon, gamma, uniform as scipy_uniform
from hyperloglog import calc_hyperloglog
import Distributaions as Distributions
import os


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

def calc_theoretical_entropy(pdf, a=0, b=1, num_points=1000):
    """
    מחשב אנטרופיה דיפרנציאלית של התפלגות עם pdf על הקטע [a,b]
    h(X) = -∫ f(x) log(f(x)) dx
    """
    # נבנה גריד לנקודות
    xs = np.linspace(a, b, num_points)
    pdf_vals = np.array([pdf(x) for x in xs])
    pdf_vals = np.clip(pdf_vals, 1e-12, None)  # למנוע log(0)
    entropy_integrand = -pdf_vals * np.log(pdf_vals)
    entropy = np.trapz(entropy_integrand, xs)
    return entropy

def calc_entropy(set1, num_bins=50):
    values = [node.value for node in set1]
    hist, _ = np.histogram(values, bins=num_bins, range=(0, 1), density=False)
    p = hist / np.sum(hist)
    p = p[p > 0]  # ignore empty bins
    entropy = -np.sum(p * np.log(p))  # ln → units: nats
    return entropy

# def calc_entropy(set1: Set[Node]) -> float:
#     # Extract the probability values from the nodes
#     probs = [node.get_probability() for node in set1]

#     # Compute the total sum of probabilities
#     total = sum(probs)

#     # If all probabilities are zero, there is no information (entropy = 0)
#     if total == 0:
#         return 0.0

#     # Normalize probabilities so they sum to 1 (required for entropy calculation)
#     normalized_probs = [p / total for p in probs]

#     # Compute entropy using the Shannon formula: H = -∑ p * log2(p)
#     entropy = -sum(p * np.log2(p) for p in normalized_probs if p > 0)

#     return entropy

def calculate_max_pdf(pdf: Callable[[float], float], resolution: int = 1000) -> float:
    x_values = np.linspace(0, 1, resolution)
    pdf_values = [pdf(x) for x in x_values]
    return max(pdf_values)


def run_simulation(set_size: int, prob_func: Callable[[float], float], max_pdf: float = 1.0,
                   num_permutations: int = 10, num_strata: int = 8,
                   distribution_name: str = "Unknown", expectation: float = 0.0, mu: float = None):
    set1 = generate_first_set(set_size, prob_func, max_pdf)
    expectation = calc_expectation(set1)
    #entropy = calc_entropy(set1)
    entropy = calc_theoretical_entropy(prob_func)
    set2 = generate_second_set(set1)

    new_set1 = {node.value for node in set1}
    new_set2 = {node.value for node in set2}
    intersection = len(new_set1 & new_set2)
    union = len(new_set1 | new_set2)
    actual_diff = union - intersection

    # Estimate symmetric difference using SOTA algorithms estimator
    symmetric_diff_minhash = calc_symmetric_diff_minhash(new_set1, new_set2)
    ##symmetric_diff_hyperloglog = calc_hyperloglog(new_set1, new_set2)
    symmetric_diff_theoretical = (1 - expectation) * len(new_set1)

    print(f"\n============== {distribution_name} Distribution ===============")
    print (f"\nExpectation {mu:.4f}")
    print(f"Set 1 size: {len(new_set1)}")
    print(f"Set 2 size: {len(new_set2)}")
    print(f"Intersection size: {intersection}")
    print(f"Union size: {union}")
    print(f"\nActual diff size: {actual_diff:.4f}")
    print(f"Expectation of symmetric diff: {symmetric_diff_theoretical:4f}")
    print(f"Entropy of Set 1: {entropy:.4f}")
    print(f"MinHash symmetric diff: {symmetric_diff_minhash:.4f}")

    # debug = [node.probability for node in set1]
    # print(f"Debug: {debug}")


    # Return results for plotting
    return [
        {
            "Distribution": distribution_name,
            "Mean": mu,
            "SetSize": set_size,
            "Entropy": entropy,
            "Algorithm": "MinHash",
            "ErrorPercent": abs(symmetric_diff_minhash - actual_diff) / actual_diff * 100 if actual_diff > 0 else 0
        },
        {
            "Distribution": distribution_name,
            "Mean": mu,
            "SetSize": set_size,
            "Entropy": entropy,
            "Algorithm": "Theoretical Expectation",
            "ErrorPercent": abs(symmetric_diff_theoretical - actual_diff) / actual_diff * 100 if actual_diff > 0 else 0
        }
    ]



def print_plots(results):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools

    # Create directory for plots if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    df = pd.DataFrame(results)

    # Plot 1: Error vs. Set Size (for each Mean and Distribution)
    for dist in df["Distribution"].unique():
        for mu in df["Mean"].unique():
            subset = df[(df["Distribution"] == dist) & (df["Mean"] == mu)]
            if subset.empty:
                continue
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=subset, x="SetSize", y="ErrorPercent", hue="Algorithm", style="Algorithm", markers=True)
            plt.title(f"Prediction Error vs. Set Size\nDistribution: {dist}, Mean: {mu}")
            plt.ylabel("Error [%]")
            plt.xlabel("Set Size")
            plt.grid(True)
            plt.savefig(f"plots/error_vs_setsize_{dist.lower()}_mean_{mu}.png")
            plt.close()

    # Plot 2: Error vs. Entropy (for each SetSize and Mean, all distributions together)
    for mu in df["Mean"].unique():
        for size in df["SetSize"].unique():
            subset = df[(df["Mean"] == mu) & (df["SetSize"] == size)]
            if subset.empty:
                continue
            plt.figure(figsize=(10, 6))
            ax = sns.scatterplot(data=subset, x="Entropy", y="ErrorPercent", hue="Algorithm", alpha=0.7)
            # Annotate each point with its distribution
            for i, row in subset.iterrows():
                plt.text(row["Entropy"], row["ErrorPercent"], row["Distribution"], fontsize=8, alpha=0.7)
            plt.title(f"Prediction Error vs. Entropy\nMean: {mu}, SetSize: {size} (All Distributions)")
            plt.ylabel("Error [%]")
            plt.xlabel("Entropy")
            plt.grid(True)
            plt.savefig(f"plots/error_vs_entropy_mean_{mu}_size_{size}.png")
            plt.close()

    # Plot 3: Error vs. Mean (for each SetSize and Distribution)
    for dist in df["Distribution"].unique():
        for size in df["SetSize"].unique():
            subset = df[(df["Distribution"] == dist) & (df["SetSize"] == size)]
            if subset.empty:
                continue
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=subset, x="Mean", y="ErrorPercent", hue="Algorithm", style="Algorithm", markers=True)
            plt.title(f"Prediction Error vs. Expectation (μ)\nDistribution: {dist}, SetSize: {size}")
            plt.ylabel("Error [%]")
            plt.xlabel("Expectation (μ)")
            plt.grid(True)
            plt.savefig(f"plots/error_vs_mean_{dist.lower()}_size_{size}.png")
            plt.close()

    # Plot 4: Error vs. Set Size for Each Distribution and Mean (already covered in Plot 1)
    # (No need to repeat)

if __name__ == "__main__":
    print("Running simulation...")
    results = []
    set_sizes = [100, 500, 1000]
    mus = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
                sim_results = run_simulation(
                    set_size=size,
                    prob_func=prob_func,
                    max_pdf=max_pdf,
                    distribution_name=dist_name,
                    mu=mu
                )
                results.extend(sim_results)
    print_plots(results)

    # Save the results DataFrame to a CSV file
    import pandas as pd
    import os
    os.makedirs("plots", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("plots/simulation_results.csv", index=False)
