import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from typing import Dict, Optional

from .distributions import mean_variance_from_spec


# ---------- Helper ----------
def r4(x: float) -> float:
    return round(float(x), 4)


# ---------- Result Model (simple) ----------
@dataclass
class AnalyticalResult:
    interarrival_rate: float  # lambda
    service_rate: float       # mu
    utilization: float        # rho
    var_services: float
    var_interarrivals: float
    Lq: float
    Wq: float
    W: float
    L: float
    note: Optional[str] = None


# ---------- M/M/1 ----------
def mm1(lambda_: float, mu: float) -> AnalyticalResult:
    if lambda_ <= 0 or mu <= 0:
        raise ValueError("lambda and mu must be > 0")

    if lambda_ >= mu:
        return AnalyticalResult(
            interarrival_rate=lambda_,
            service_rate=mu,
            utilization=r4(lambda_ / mu),
            var_services=r4(1.0 / (mu * mu)),
            var_interarrivals=r4(1.0 / (lambda_ * lambda_)),
            Lq=float("inf"),
            Wq=float("inf"),
            W=float("inf"),
            L=float("inf"),
            note="Unstable system (λ ≥ μ)"
        )

    rho = lambda_ / mu
    Lq = (rho * rho) / (1.0 - rho)
    Wq = Lq / lambda_
    W = Wq + (1.0 / mu)
    L = lambda_ * W

    return AnalyticalResult(
        interarrival_rate=r4(lambda_),
        service_rate=r4(mu),
        utilization=r4(rho),
        var_services=r4(1.0 / (mu * mu)),
        var_interarrivals=r4(1.0 / (lambda_ * lambda_)),
        Lq=r4(Lq),
        Wq=r4(Wq),
        W=r4(W),
        L=r4(L),
        note=None
    )


# ---------- Erlang C helpers for M/M/c ----------
def _erlang_c(lambda_: float, mu: float, c: int) -> Tuple[float, float]:
    """
    Returns (Pw, rho) where:
    rho = lambda / (c*mu)
    Pw = probability that an arrival must wait (Erlang C)
    """
    if c <= 0:
        raise ValueError("c must be >= 1")

    rho = lambda_ / (c * mu)
    if rho >= 1:
        return float("inf"), rho

    a = lambda_ / mu  # offered load

    # sum_{n=0}^{c-1} (a^n / n!)
    s = 0.0
    for n in range(c):
        s += (a ** n) / math.factorial(n)

    # (a^c / c!) * (c / (c - a))
    last = (a ** c) / math.factorial(c) * (c / (c - a))

    P0 = 1.0 / (s + last)
    Pw = last * P0
    return Pw, rho


# ---------- M/M/c ----------
def mmc(lambda_: float, mu: float, c: int) -> AnalyticalResult:
    if lambda_ <= 0 or mu <= 0:
        raise ValueError("lambda and mu must be > 0")
    if c <= 0:
        raise ValueError("servers c must be >= 1")

    Pw, rho = _erlang_c(lambda_, mu, c)

    # Variances for M/M/c (both exponential)
    varA = 1.0 / (lambda_ * lambda_)
    varS = 1.0 / (mu * mu)

    if rho == float("inf") or rho >= 1:
        return AnalyticalResult(
            interarrival_rate=r4(lambda_),
            service_rate=r4(mu),
            utilization=r4(lambda_ / (c * mu)),
            var_services=r4(varS),
            var_interarrivals=r4(varA),
            Lq=float("inf"),
            Wq=float("inf"),
            W=float("inf"),
            L=float("inf"),
            note="Unstable system (λ ≥ cμ)"
        )

    # Standard M/M/c results:
    # Wq = Pw / (c*mu - lambda)
    Wq = Pw / (c * mu - lambda_)
    Lq = lambda_ * Wq
    W = Wq + (1.0 / mu)
    L = lambda_ * W

    return AnalyticalResult(
        interarrival_rate=r4(lambda_),
        service_rate=r4(mu),
        utilization=r4(rho),
        var_services=r4(varS),
        var_interarrivals=r4(varA),
        Lq=r4(Lq),
        Wq=r4(Wq),
        W=r4(W),
        L=r4(L),
        note=None
    )
# import math
# from typing import Dict, Optional

# ---------- Utility ----------
def _scv(var: float, mean: float) -> float:
    # squared coefficient of variation
    if mean <= 0:
        return 0.0
    return var / (mean * mean)

# ---------- M/G/1 (Pollaczek–Khinchine) ----------
def mg1(lambda_: float, mu: float, varS: float) -> AnalyticalResult:
    """
    Uses:
      E[S] = 1/mu
      rho = lambda * E[S] = lambda/mu
      Wq = (lambda * E[S^2]) / (2*(1-rho))
      where E[S^2] = Var(S) + (E[S])^2
      Lq = lambda * Wq
      W = Wq + E[S]
      L = lambda * W
    """
    if lambda_ <= 0 or mu <= 0:
        raise ValueError("lambda and mu must be > 0")

    ES = 1.0 / mu
    rho = lambda_ * ES

    # arrival is exponential in M/G/1:
    varA = 1.0 / (lambda_ * lambda_)

    if rho >= 1:
        return AnalyticalResult(
            interarrival_rate=r4(lambda_),
            service_rate=r4(mu),
            utilization=r4(rho),
            var_services=r4(varS),
            var_interarrivals=r4(varA),
            Lq=float("inf"), Wq=float("inf"), W=float("inf"), L=float("inf"),
            note="Unstable system (ρ ≥ 1)"
        )

    ES2 = varS + ES * ES
    Wq = (lambda_ * ES2) / (2.0 * (1.0 - rho))
    Lq = lambda_ * Wq
    W = Wq + ES
    L = lambda_ * W

    return AnalyticalResult(
        interarrival_rate=r4(lambda_),
        service_rate=r4(mu),
        utilization=r4(rho),
        var_services=r4(varS),
        var_interarrivals=r4(varA),
        Lq=r4(Lq),
        Wq=r4(Wq),
        W=r4(W),
        L=r4(L),
        note=None
    )

# ---------- G/G/1 (Kingman approximation) ----------
def gg1(lambda_: float, mu: float, varA: float, varS: float) -> AnalyticalResult:
    """
    Kingman's approximation:
      Wq ≈ ( (Ca^2 + Cs^2) / 2 ) * ( ρ / (1-ρ) ) * E[S]
    where:
      Ca^2 = Var(A) / (E[A])^2, E[A]=1/lambda
      Cs^2 = Var(S) / (E[S])^2, E[S]=1/mu
    """
    if lambda_ <= 0 or mu <= 0:
        raise ValueError("lambda and mu must be > 0")

    EA = 1.0 / lambda_
    ES = 1.0 / mu
    rho = lambda_ * ES

    if rho >= 1:
        return AnalyticalResult(
            interarrival_rate=r4(lambda_),
            service_rate=r4(mu),
            utilization=r4(rho),
            var_services=r4(varS),
            var_interarrivals=r4(varA),
            Lq=float("inf"), Wq=float("inf"), W=float("inf"), L=float("inf"),
            note="Unstable system (ρ ≥ 1)"
        )

    Ca2 = _scv(varA, EA)
    Cs2 = _scv(varS, ES)

    Wq = ((Ca2 + Cs2) / 2.0) * (rho / (1.0 - rho)) * ES
    Lq = lambda_ * Wq
    W = Wq + ES
    L = lambda_ * W

    return AnalyticalResult(
        interarrival_rate=r4(lambda_),
        service_rate=r4(mu),
        utilization=r4(rho),
        var_services=r4(varS),
        var_interarrivals=r4(varA),
        Lq=r4(Lq),
        Wq=r4(Wq),
        W=r4(W),
        L=r4(L),
        note=None
    )

# ---------- G/G/c (Allen–Cunneen approximation) ----------
def ggc(lambda_: float, mu: float, c: int, varA: float, varS: float) -> AnalyticalResult:
    """
    Allen–Cunneen style:
      Wq ≈ ( (Ca^2 + Cs^2) / 2 ) * (Pw / (c*mu - lambda))   [scaled using Erlang C Pw]
    We use Erlang C waiting probability computed as if M/M/c (same lambda/mu/c),
    then scale by variability term (Ca^2 + Cs^2)/2.
    """
    if lambda_ <= 0 or mu <= 0:
        raise ValueError("lambda and mu must be > 0")
    if c <= 0:
        raise ValueError("servers c must be >= 1")

    EA = 1.0 / lambda_
    ES = 1.0 / mu

    Ca2 = _scv(varA, EA)
    Cs2 = _scv(varS, ES)

    Pw, rho = _erlang_c(lambda_, mu, c)

    if rho == float("inf") or rho >= 1:
        return AnalyticalResult(
            interarrival_rate=r4(lambda_),
            service_rate=r4(mu),
            utilization=r4(lambda_ / (c * mu)),
            var_services=r4(varS),
            var_interarrivals=r4(varA),
            Lq=float("inf"), Wq=float("inf"), W=float("inf"), L=float("inf"),
            note="Unstable system (λ ≥ cμ)"
        )

    base_Wq = Pw / (c * mu - lambda_)  # M/M/c Wq
    Wq = ((Ca2 + Cs2) / 2.0) * base_Wq

    Lq = lambda_ * Wq
    W = Wq + ES
    L = lambda_ * W

    return AnalyticalResult(
        interarrival_rate=r4(lambda_),
        service_rate=r4(mu),
        utilization=r4(rho),
        var_services=r4(varS),
        var_interarrivals=r4(varA),
        Lq=r4(Lq),
        Wq=r4(Wq),
        W=r4(W),
        L=r4(L),
        note=None
    )

# ---------- General solver (based on chosen model) ----------
def solve_analytical(model: str, lambda_: float, mu: float, c: int,
                     arrival_spec: Optional[Dict] = None,
                     service_spec: Optional[Dict] = None) -> AnalyticalResult:
    """
    Supported:
      - M/M/1
      - M/M/c
      - M/G/1
      - G/G/1
      - G/G/c
    arrival_spec/service_spec are dicts like:
      {"dist_type": "gamma", "params": {"shape":2, "scale":1}}
    """

    model = model.strip().upper()

    # default for "M": exponential interarrival/service with rates lambda, mu
    if arrival_spec is None:
        arrival_spec = {"dist_type": "exponential", "params": {"rate": lambda_}}
    if service_spec is None:
        service_spec = {"dist_type": "exponential", "params": {"rate": mu}}

    meanA, varA = mean_variance_from_spec(arrival_spec)
    meanS, varS = mean_variance_from_spec(service_spec)

    if model in ["M/M/1", "MM1"]:
        return mm1(lambda_, mu)

    if model in ["M/M/C", "M/M/C".upper(), "MM/C".upper(), "MMC"]:
        return mmc(lambda_, mu, c)

    if model in ["M/G/1", "MG1"]:
        # arrival is exponential, service is general (varS from spec)
        return mg1(lambda_, mu, varS)

    if model in ["G/G/1", "GG1"]:
        return gg1(lambda_, mu, varA, varS)

    if model in ["G/G/C", "G/G/C".upper(), "GGC"]:
        return ggc(lambda_, mu, c, varA, varS)

    raise NotImplementedError(f"Model not implemented yet: {model}")
