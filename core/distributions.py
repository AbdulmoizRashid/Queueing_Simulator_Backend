import random
import math
from typing import Tuple, Dict

def sample_exponential(rate: float, rng: random.Random) -> float:
    # mean = 1/rate
    return -math.log(1.0 - rng.random()) / rate

def sample_uniform(a: float, b: float, rng: random.Random) -> float:
    return rng.uniform(a, b)

def sample_normal(mean: float, std: float, rng: random.Random) -> float:
    # truncate at 0 to avoid negative times
    x = rng.gauss(mean, std)
    return x if x > 0 else 0.0

def sample_gamma(shape: float, scale: float, rng: random.Random) -> float:
    # Python's random.gammavariate(alpha, beta) => alpha=shape, beta=scale
    return rng.gammavariate(shape, scale)

def sample_from_spec(spec: Dict, rng: random.Random) -> Tuple[float, float, str]:
    """
    Returns: (value, cp, cp_lookup_string)
    cp is the raw U(0,1) used for traceability (good for your CP column).
    """
    u = rng.random()
    dist_type = spec["dist_type"].strip().lower()
    p = spec["params"]

    if dist_type == "exponential":
        rate = p["rate"]
        value = -math.log(1.0 - u) / rate
        lookup = "InverseExp(U)"
    elif dist_type == "uniform":
        a, b = p["min"], p["max"]
        value = a + (b - a) * u
        lookup = "Uniform[min,max]"
    elif dist_type == "normal":
        # can't invert easily; we use gauss separately, cp is still shown
        value = sample_normal(p["mean"], p["std"], rng)
        lookup = "Normal(mean,std)"
    elif dist_type == "gamma":
        value = sample_gamma(p["shape"], p["scale"], rng)
        lookup = "Gamma(shape,scale)"
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    return value, u, lookup

def mean_variance_from_spec(spec: Dict) -> Tuple[float, float]:
    dist_type = spec["dist_type"].strip().lower()
    p = spec["params"]

    if dist_type == "exponential":
        rate = p["rate"]
        mean = 1.0 / rate
        var = 1.0 / (rate * rate)
    elif dist_type == "uniform":
        a, b = p["min"], p["max"]
        mean = (a + b) / 2.0
        var = ((b - a) ** 2) / 12.0
    elif dist_type == "normal":
        mean = p["mean"]
        var = p["std"] ** 2
    elif dist_type == "gamma":
        k, theta = p["shape"], p["scale"]
        mean = k * theta
        var = k * (theta ** 2)
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    return mean, var
