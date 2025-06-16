from datasketch import HyperLogLog

data1 = ['hyperloglog', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
'estimating', 'the', 'cardinality', 'of', 'dataset', 'dataset', 'a']

def calc_hyperloglog(set_a: set, set_b: set):

  hll_a = HyperLogLog(p=14)
  hll_b = HyperLogLog(p=14)

  for item in set_a:
    hll_a.update(str(item).encode('utf8'))

  for item in set_b:
    hll_b.update(str(item).encode('utf8'))

  est_a = len(hll_a)
  est_b = len(hll_b)

  # Merge hll_b into hll_a
  hll_a.merge(hll_b)
  est_union = len(hll_a)

  est_intersection = est_a + est_b - est_union

  # |A| + |B| - 2 * |A ∩ B|
  est_symmetric_diff = est_a + est_b - 2 * est_intersection

  return est_symmetric_diff