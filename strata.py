import hashlib
import math
from collections import defaultdict

def hash_item(item, bits=64):
    h = hashlib.sha256(str(item).encode()).hexdigest()
    return int(h, 16) & ((1 << bits) - 1)

def leading_zeros(n, bits=64):
    return bits - n.bit_length() if n > 0 else bits

def build_strata_set(data, max_strata=64):
    strata = defaultdict(list)
    for item in data:
        hashed = hash_item(item)
        lz = leading_zeros(hashed, bits=64)
        if lz < max_strata:
            strata[lz].append(hashed)
    return strata

def estimate_symmetric_difference_strata(A, B, max_strata=64):
    strata_A = build_strata_set(A, max_strata)
    strata_B = build_strata_set(B, max_strata)

    estimate = 0
    for stratum in range(max_strata):
        a_set = set(strata_A.get(stratum, []))
        b_set = set(strata_B.get(stratum, []))

        unique = len(a_set.symmetric_difference(b_set))
        if unique > 0:
            estimate = unique * (2 ** stratum)
            break

    return estimate