import hashlib
from Py_IBLT import IBLT
from typing import Set

def count_leading_zeros(data: bytes) -> int:
    bits = ''.join(f"{byte:08b}" for byte in data)
    return len(bits) - len(bits.lstrip('0'))

def hash_to_stratum(x: str) -> int:
    h = hashlib.sha256(str(x).encode('utf8')).digest()
    return count_leading_zeros(h)

def estimate_sym_diff_strata(
    set_a: Set[str],
    set_b: Set[str],
    max_strata: int = 64,
    iblt_size: int = 1024,
    num_hashes: int = 3
) -> int:
    # Create one IBLT per stratum for each set
    iblt_a = [IBLT(iblt_size, num_hashes, key_size=0, value_size=0) for _ in range(max_strata)]
    iblt_b = [IBLT(iblt_size, num_hashes, key_size=0, value_size=0) for _ in range(max_strata)]

    # Insert elements into corresponding IBLTs
    for x in set_a:
        s = hash_to_stratum(x)
        if s >= max_strata: continue
        iblt_a[s].insert(x, None)

    for x in set_b:
        s = hash_to_stratum(x)
        if s >= max_strata: continue
        iblt_b[s].insert(x, None)

    # Find highest stratum that decodes successfully
    highest = -1
    for s in range(max_strata):
        diff_iblt = iblt_a[s].subtract(iblt_b[s])
        status, entries, deleted = diff_iblt.list_entries()
        # RESULT_LIST_ENTRIES_COMPLETE indicates full decode
        if status == IBLT.RESULT_LIST_ENTRIES_COMPLETE:
            highest = s

    if highest == -1:
        raise RuntimeError("No decodable strata; try increasing IBLT size.")

    # Estimate symmetric difference as approx 2^highest
    return 2 ** highest

# Example usage:

