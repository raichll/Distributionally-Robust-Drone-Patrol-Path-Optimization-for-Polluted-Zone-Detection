import scipy as sp
import numpy as np
import random
import math


class RoundHash(object):
    """
    Create a 'hash' function based on rounding vectors.
    """

    def __init__(self, bucket_size):
        self.bucket_size = bucket_size

    def apply(self, x):
        x = np.array(x) / self.bucket_size
        x = np.floor(x)
        return hash(x.data.tobytes())



class LSHash(object):
    """Create a single lsh function as described in Lemma 2 of the paper."""

    def __init__(self, bucket_size):
        self.round_hash = RoundHash(bucket_size)
        self.eta = random.uniform(0, bucket_size)

    def apply(self, x):
        x = np.array(x) + self.eta * np.ones(x.shape)
        return self.round_hash.apply(x)