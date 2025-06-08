import numpy as np
from typing import Set, List, Any

class MinHash:
    def __init__(self, num_permutations: int = 5):
        """
        Initialize MinHash with a specified number of permutations.
        
        Args:
            num_permutations (int): Number of hash functions to use
        """
        self.num_permutations = num_permutations
        # Generate random hash functions parameters (ax + b mod p)
        self.prime = 2**31 - 1  # Large prime number
        self.a = np.random.randint(1, self.prime, num_permutations)
        self.b = np.random.randint(0, self.prime, num_permutations)
    
    def _hash_function(self, x: int, a: int, b: int) -> int:
        """Apply a hash function of the form (ax + b) mod p."""
        return ((a * x + b) % self.prime)
    
    def get_signature(self, input_set: Set[Any]) -> List[int]:
        """
        Compute MinHash signature for a set.
        
        Args:
            input_set: Input set of hashable elements
        
        Returns:
            List of minimum hash values (signature)
        """
        # Convert set elements to integers using hash()
        set_hashes = [hash(x) for x in input_set]
        
        signature = []
        for i in range(self.num_permutations):
            # Apply the hash function to all elements and take minimum
            min_hash = min(self._hash_function(x, self.a[i], self.b[i]) 
                         for x in set_hashes)
            signature.append(min_hash)
        
        return signature
    
    def estimate_similarity(self, set1: Set[Any], set2: Set[Any]) -> float:
        """
        Estimate Jaccard similarity between two sets using MinHash.
        
        Args:
            set1: First input set
            set2: Second input set
        
        Returns:
            Estimated Jaccard similarity [0,1]
        """
        sig1 = self.get_signature(set1)
        sig2 = self.get_signature(set2)
        
        # Count matching elements in signatures
        matches = sum(1 for i in range(self.num_permutations) 
                     if sig1[i] == sig2[i])
        
        # Return estimated similarity
        return matches / self.num_permutations 