from datasketch import MinHash
from math import log


def calc_minhash(k: int, set_a: set, set_b: set):
    mh_a = MinHash(num_perm=k)
    mh_b = MinHash(num_perm=k)

    # Add the elements of the sets to the MinHash objects
    for item in set_a:
        mh_a.update(str(item).encode('utf8'))  # Convert item to string before encoding

    for item in set_b:
        mh_b.update(str(item).encode('utf8'))  # Convert item to string before encoding

    # # Print the MinHash values
    # print(mh_a.digest())
    # print(mh_b.digest())
    #
    # # Estimate the similarity between the sets
    similarity = mh_a.jaccard(mh_b)
    # print(similarity)

    return similarity


def calc_symmetric_diff_minhash(set_a: set, set_b: set):
    #k = int(log(len(set_a)))
    k = 10
    J = calc_minhash(k, set_a, set_b)
    # return J
    return (len(set_a) + len(set_b)) * ((1 - J) / (1 + J))
