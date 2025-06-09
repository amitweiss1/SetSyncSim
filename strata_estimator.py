import numpy as np
from typing import Set, Any, Tuple

class StrataEstimator:
    def __init__(self, num_strata: int = 8, hash_range: int = 2**32 - 1):
        self.num_strata = num_strata
        self.hash_range = hash_range

    def _stratum(self, hash_val: int) -> int:
        """
        Determine the stratum index based on the leading zero bits.
        """
        # Calculate how many leading zeros in binary representation
        for i in range(self.num_strata):
            if hash_val < (1 << (self.hash_range.bit_length() - i - 1)):
                return i
        return self.num_strata - 1

    def _stratify(self, input_set: Set[Any]) -> dict:
        """
        Assign elements to strata based on their hash value.
        """
        strata = {i: set() for i in range(self.num_strata)}
        for element in input_set:
            h = hash(element) % self.hash_range
            s = self._stratum(h)
            strata[s].add(h)
        return strata

    def estimate_similarity(self, set1: Set[Any], set2: Set[Any]) -> Tuple[float, float]:
        """
        Estimate Jaccard similarity using the strata method.

        Returns:
            Tuple of (estimated_similarity, estimated_error)
        """
        strata1 = self._stratify(set1)
        strata2 = self._stratify(set2)

        total_matches = 0
        total_candidates = 0

        for i in range(self.num_strata):
            h1 = strata1[i]
            h2 = strata2[i]
            intersection = len(h1 & h2)
            union = len(h1 | h2)
            if union > 0:
                total_matches += intersection
                total_candidates += union

        if total_candidates == 0:
            return 0.0, 0.0

        estimate = total_matches / total_candidates
        error = 1 / np.sqrt(total_candidates) if total_candidates > 0 else 1.0
        return estimate, error
