import numpy as np
from typing import Set, Callable
from node import Node
from minhash import MinHash
from strata_estimator import StrataEstimator

from scipy.stats import norm, uniform, expon, gamma
from typing import Callable


def sample_from_pdf(pdf: Callable[[float], float], max_pdf: float) -> float:
    while True:
        x = round(np.random.uniform(0, 1), 3)
        y = np.random.uniform(0, max_pdf)
        if y <= pdf(x):
            return x


def generate_first_set(size: int, prob_func: Callable[[float], float], max_pdf) -> Set[Node]:
    """
    Generate a set of integers based on a probability function.
    
    Args:
        size (int): Desired size of the set
        prob_func (Callable): Function that returns probability of including each number
    
    Returns:
        Set of integers
    """
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


class ProbabilityDistributions:

    @staticmethod
    def normal(mean=0.5, std=0.1) -> Callable[[float], float]:
        dist = norm(loc=mean, scale=std)
        norm_const = dist.cdf(1) - dist.cdf(0)
        return lambda x: dist.pdf(x) / norm_const if 0 <= x <= 1 else 0

    @staticmethod
    def uniform(a=0, b=1) -> Callable[[float], float]:
        dist = uniform(loc=a, scale=b - a)
        return lambda x: dist.pdf(x) if a <= x <= b else 0

    @staticmethod
    def exponential(lambda_param=1) -> Callable[[float], float]:
        dist = expon(scale=1 / lambda_param)
        norm_const = dist.cdf(1) - dist.cdf(0)
        return lambda x: dist.pdf(x) / norm_const if 0 <= x <= 1 else 0

    @staticmethod
    def gamma(alpha=2, beta=1) -> Callable[[float], float]:
        dist = gamma(a=alpha, scale=1 / beta)
        norm_const = dist.cdf(1) - dist.cdf(0)
        return lambda x: dist.pdf(x) / norm_const if 0 <= x <= 1 else 0


def run_simulation(set_size: int, prob_func: Callable[[int], float], max_pdf: float = 1.0,
                   num_permutations: int = 10, num_strata: int = 8,
                   distribution_name: str = "Unknown"):
    """
    Run simulation comparing MinHash and Strata Estimator.

    Args:
        set_size (int): Size of sets to generate
        prob_func (Callable): Probability function for set generation
        num_permutations (int): Number of permutations for MinHash
        num_strata (int): Number of strata for Strata Estimator
        distribution_name (str): Name of the probability distribution
    """
    # Generate two sets
    set1 = generate_first_set(set_size, prob_func, max_pdf)
    set2 = generate_second_set(set1)

    new_set1 = set()
    new_set2 = set()
    for node in set1:
        new_set1.add(node.value)
    for node in set2:
        new_set2.add(node.value)

    # print(f"\n=== {distribution_name} Distribution ===")
    # print(set1)
    # print("size of set1:", len(set1))
    # print("=" * 50)
    # print(set2)
    # print("size of set2:", len(set2))

    # # Calculate actual Jaccard similarity
    intersection = len(new_set1 & new_set2)
    union = len(new_set1 | new_set2)
    actual_diff = union - intersection
    #
    # # MinHash estimation
    minhash = MinHash(num_permutations=num_permutations)
    minhash_similarity = minhash.estimate_similarity(new_set1, new_set2)
    #
    # Strata estimation
    strata = StrataEstimator(num_strata=num_strata)
    strata_similarity, strata_error = strata.estimate_similarity(new_set1,new_set2)
    #
    # Print results
    print(f"\n=== {distribution_name} Distribution ===============")
    print(f"Set 1 size: {len(new_set1)}")
    print(f"Set 2 size: {len(new_set2)}")
    print(f"Intersection size: {intersection}")
    print(f"Union size: {union}")
    print(f"\nActual diff size: {(actual_diff):.4f}")
    print(f"MinHash estimated diff: {(1-minhash_similarity):.4f}")

    print(f"Strata estimated diff: {(1 - strata_similarity):.4f}")
    print("=" * 50)

if __name__ == "__main__":
    print("Running simulation...")
    set_sizes = [10000]

    # Create instances of all distributions
    distributions = {
        "Normal": {
            "func": ProbabilityDistributions.normal(mean=0.5, std=0.1),
            "max_pdf": 4.0
        },
        "Uniform": {
            "func": ProbabilityDistributions.uniform(0, 1),
            "max_pdf": 1.0
        },
        "Exponential": {
            "func": ProbabilityDistributions.exponential(lambda_param=5),
            "max_pdf": 2.5
        },
        "Gamma": {
            "func": ProbabilityDistributions.gamma(alpha=2, beta=2),
            "max_pdf": 2.0
        }
    }

    # Run simulation for each distribution
    for size in set_sizes:
        for dist_name, dist_config in distributions.items():
            prob_func = dist_config["func"]
            max_pdf = dist_config["max_pdf"]
            run_simulation(
                set_size=size,
                prob_func=prob_func,
                max_pdf=max_pdf,
                distribution_name=dist_name
            )