import numpy as np
from typing import Set, Callable
from node import Node
from minhash import MinHash
from strata_estimator import  estimate_sym_diff_strata
from minhash_default import calc_symmetric_diff_minhash
from scipy.stats import norm, uniform, expon, gamma
from hyperloglog import calc_hyperloglog


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
        dist = uniform(loc=a, scale=b - a)
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


def generate_first_set(size: int, prob_func: Callable[[float], float], max_pdf) -> Set[Node]:
    result = set()
    for index in range(1, size + 1):
        new_node = Node(index)
        new_node.set_probability(sample_from_pdf(prob_func, max_pdf))
        result.add(new_node)
    return result

def calc_expectation(set1: Set[Node]) -> float:
    total = sum(node.get_probability() for node in set1)
    return total / len(set1) if set1 else 0.0

def generate_second_set(set1: Set[Node]) -> Set[Node]:
    set2 = set()
    for node in set1:
        new_node = Node(node.value)
        x = round(np.random.uniform(0, 1), 3)
        if x <= node.get_probability():
            set2.add(new_node)
    return set2


def run_simulation(set_size: int, prob_func: Callable[[int], float], max_pdf: float = 1.0,
                   num_permutations: int = 10, num_strata: int = 8,
                   distribution_name: str = "Unknown", expectation: float = 0.0):
    set1 = generate_first_set(set_size, prob_func, max_pdf)
    expectation = calc_expectation(set1)
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
    print(f"Expectation of symmetric diff: {(1 - expectation)*len(new_set1):4f}")
    print(f"MinHash symmetric diff: {symmetric_diff_minhash:.4f}")
    print(f"HyperLogLog symmetric diff: {symmetric_diff_hyperloglog:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    print("Running simulation...")
    set_sizes = [10000]

    distributions = {
        "Normal": ProbabilityDistributions.normal(mean=0.5, std=0.1),
        "Uniform": ProbabilityDistributions.uniform(a=0, b=1),
        "Exponential": ProbabilityDistributions.exponential(lambda_param=5),
        "Gamma": ProbabilityDistributions.gamma(alpha=2, beta=1)
    }

    for size in set_sizes:
        for dist_name, dist_config in distributions.items():
            prob_func = dist_config.pdf
            max_pdf = dist_config.expectation
            run_simulation(
                set_size=size,
                prob_func=prob_func,
                max_pdf=max_pdf,
                distribution_name=dist_name
            )