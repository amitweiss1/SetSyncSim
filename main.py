import numpy as np
from typing import Set, Callable
from minhash import MinHash
from strata_estimator import StrataEstimator
from scipy import stats

def generate_set(size: int, prob_func: Callable[[int], float]) -> Set[int]:
    """
    Generate a set of integers based on a probability function.
    
    Args:
        size (int): Desired size of the set
        prob_func (Callable): Function that returns probability of including each number
    
    Returns:
        Set of integers
    """
    result = set()
    num = 0
    while len(result) < size:
        if np.random.random() < prob_func(num):
            result.add(num)
        num += 1
    return result

class ProbabilityDistributions:
    @staticmethod
    def normal(mean=0, std=1):
        """Normal distribution probability function."""
        return lambda x: max(0, min(1, stats.norm.pdf(x, mean, std) * 5))
    
    @staticmethod
    def binomial(n=10, p=0.5):
        """Binomial distribution probability function."""
        return lambda x: max(0, min(1, stats.binom.pmf(x % (n+1), n, p) * 3))
    
    @staticmethod
    def poisson(lambda_param=5):
        """Poisson distribution probability function."""
        return lambda x: max(0, min(1, stats.poisson.pmf(x, lambda_param) * 3))
    
    @staticmethod
    def uniform(a=0, b=1):
        """Uniform distribution probability function."""
        return lambda x: b if a <= x <= b else 0
    
    @staticmethod
    def exponential(lambda_param=1):
        """Exponential distribution probability function."""
        return lambda x: max(0, min(1, stats.expon.pdf(x, scale=1/lambda_param) * 3))
    
    @staticmethod
    def gamma(alpha=2, beta=1):
        """Gamma distribution probability function."""
        return lambda x: max(0, min(1, stats.gamma.pdf(x, alpha, scale=1/beta) * 3))
    
    @staticmethod
    def geometric(p=0.5):
        """Geometric distribution probability function."""
        return lambda x: max(0, min(1, stats.geom.pmf(x+1, p) * 3))

def run_simulation(set_size: int, prob_func: Callable[[int], float], 
                  num_permutations: int = 100, num_strata: int = 8,
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
    set1 = generate_set(set_size, prob_func)
    set2 = generate_set(set_size, prob_func)
    
    # Calculate actual Jaccard similarity
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    actual_similarity = intersection / union
    
    # MinHash estimation
    minhash = MinHash(num_permutations=num_permutations)
    minhash_similarity = minhash.estimate_similarity(set1, set2)
    
    # Strata estimation
    strata = StrataEstimator(num_strata=num_strata)
    strata_similarity, strata_error = strata.estimate_similarity(set1, set2)
    
    # Print results
    print(f"\n=== {distribution_name} Distribution ===")
    print(f"Set 1 size: {len(set1)}")
    print(f"Set 2 size: {len(set2)}")
    print(f"Intersection size: {intersection}")
    print(f"Union size: {union}")
    print(f"\nActual Jaccard similarity: {actual_similarity:.4f}")
    print(f"MinHash estimated similarity: {minhash_similarity:.4f}")
    print(f"Strata estimated similarity: {strata_similarity:.4f} ± {strata_error:.4f}")
    print(f"\nMinHash absolute error: {abs(minhash_similarity - actual_similarity):.4f}")
    print(f"Strata absolute error: {abs(strata_similarity - actual_similarity):.4f}")
    print("=" * 50)

if __name__ == "__main__":
    print("Running simulation...")
    # Example usage
    set_size = [10]
    
    # Create instances of all distributions
    distributions = {
        "Normal": ProbabilityDistributions.normal(mean=5, std=2),
        "Binomial": ProbabilityDistributions.binomial(n=10, p=0.5),
        "Poisson": ProbabilityDistributions.poisson(lambda_param=5),
        "Uniform": ProbabilityDistributions.uniform(a=0, b=0.5),
        "Exponential": ProbabilityDistributions.exponential(lambda_param=1),
        "Gamma": ProbabilityDistributions.gamma(alpha=2, beta=1),
        "Geometric": ProbabilityDistributions.geometric(p=0.3)
    }
    
    # Run simulation for each distribution
    for size in set_size:
        for dist_name, prob_func in distributions.items():
            run_simulation(size, prob_func, distribution_name=dist_name) 