import numpy as np
from typing import Set, List, Any, Tuple
import mmh3  # MurmurHash3 for better hash distribution

class StrataEstimator:
    def __init__(self, num_strata: int = 4, strata_size: int = 8):
        """
        Initialize Strata Estimator.
        
        Args:
            num_strata (int): Number of strata to use
            strata_size (int): Size of each stratum
        """
        self.num_strata = num_strata
        self.strata_size = strata_size
        self.total_hashes = num_strata * strata_size
    
    def _hash_element(self, element: Any, seed: int) -> int:
        """Generate hash value for an element using MurmurHash3."""
        return mmh3.hash(str(element), seed)
    
    def _get_strata(self, input_set: Set[Any]) -> List[List[int]]:
        """
        Divide elements into strata based on their hash values.
        
        Args:
            input_set: Input set of elements
        
        Returns:
            List of strata, where each stratum contains hash values
        """
        strata = [[] for _ in range(self.num_strata)]
        
        for element in input_set:
            for i in range(self.total_hashes):
                hash_value = self._hash_element(element, i)
                stratum_idx = i // self.strata_size
                strata[stratum_idx].append(hash_value)
        
        # Sort each stratum
        for stratum in strata:
            stratum.sort()
            # Keep only the smallest values up to strata_size
            del stratum[self.strata_size:]
        
        return strata
    
    def estimate_similarity(self, set1: Set[Any], set2: Set[Any]) -> Tuple[float, float]:
        """
        Estimate Jaccard similarity between two sets using Strata Estimator.
        
        Args:
            set1: First input set
            set2: Second input set
        
        Returns:
            Tuple of (estimated Jaccard similarity, estimated standard error)
        """
        strata1 = self._get_strata(set1)
        strata2 = self._get_strata(set2)
        
        # Calculate similarities for each stratum
        stratum_similarities = []
        for s1, s2 in zip(strata1, strata2):
            # Count matching elements
            matches = len(set(s1) & set(s2))
            total = len(set(s1) | set(s2))
            if total > 0:
                stratum_similarities.append(matches / total)
            else:
                stratum_similarities.append(1.0)  # Empty sets are considered identical
        
        # Calculate mean and standard error
        similarity = np.mean(stratum_similarities)
        std_error = np.std(stratum_similarities) / np.sqrt(self.num_strata)
        
        return similarity, std_error 