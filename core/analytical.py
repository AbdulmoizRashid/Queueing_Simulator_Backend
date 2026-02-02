import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from typing import Dict, Optional
from typing import Literal

from .distributions import mean_variance_from_spec


# ---------- Helper ----------
def r4(x: float) -> float:
    if x == float("inf"):
        return x
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
            interarrival_rate=r4(lambda_),
            service_rate=r4(mu),
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
# M/M/1 checked!!

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
#M/M/C Checked!!

# import math
# from typing import Dict, Optional

# ---------- Utility ----------
def _scv(var: float, mean: float) -> float:
    # squared coefficient of variation
    if mean <= 0:
        return 0.0
    return var / (mean * mean)

# ---------- M/G/1 (Pollaczek–Khinchine) ----------
ServiceType = Literal["uniform", "normal", "gamma"]

def mg1(lambda_: float, service_type: ServiceType, params: Dict) -> AnalyticalResult:
    """
    M/G/1 (Pollaczek–Khinchine)
    Arrivals are Poisson with rate λ => interarrivals exponential, Var(A)=1/λ²
    Service time is General => computed from selected distribution (uniform/normal/gamma)
    """

    if lambda_ <= 0:
        raise ValueError("lambda must be > 0")

    # M arrivals: exponential interarrivals
    varA = 1.0 / (lambda_ * lambda_)

    note = None

    # --- service moments ---
    if service_type == "uniform":
        a = params.get("min")
        b = params.get("max")
        if a is None or b is None:
            raise ValueError("Uniform service requires params: {'min': a, 'max': b}")
        a = float(a); b = float(b)
        if a < 0 or b <= a:
            raise ValueError("Uniform requires 0 <= min < max")

        ES = (a + b) / 2.0
        varS = ((b - a) ** 2) / 12.0

    elif service_type == "normal":
        m = params.get("mean")
        sd = params.get("std")
        if m is None or sd is None:
            raise ValueError("Normal service requires params: {'mean': m, 'std': std}")
        m = float(m); sd = float(sd)
        if m <= 0 or sd < 0:
            raise ValueError("Normal requires mean > 0 and sd >= 0")

        ES = m
        varS = sd * sd

        if sd > 0 and (m / sd) < 3:
            note = "Warning: Normal service may imply negative service times; consider gamma or truncated normal."


    elif service_type == "gamma":

        shape = params.get("shape")

        scale = params.get("scale")

        if shape is None or scale is None:
            raise ValueError("Gamma service requires params: {'shape': shape, 'scale': scale}")

        shape = float(shape)

        scale = float(scale)

        if shape <= 0 or scale <= 0:
            raise ValueError("Gamma requires shape > 0 and scale > 0")

        ES = shape * scale

        varS = shape * (scale ** 2)


    else:
        raise ValueError("service_type must be one of: 'uniform', 'normal', 'gamma'")

    if ES <= 0:
        raise ValueError("Computed mean service time E[S] must be > 0")

    mu = 1.0 / ES
    rho = lambda_ * ES  # = lambda/mu

    if rho >= 1.0:
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
        note=note
    )

#M/G/1 checked!!

# ---------- M/G/c (Allen–Cunneen approximation) ----------
def mgc(lambda_: float, mu: float, c: int, varS: float) -> AnalyticalResult:
    """
    M/G/c using Allen–Cunneen approximation:

      Wq(M/G/c) ≈ ((1 + Cs^2) / 2) * Wq(M/M/c)

    where:
      E[S]  = 1/mu
      Cs^2  = Var(S) / (E[S])^2
      Wq(M/M/c) = Pw / (c*mu - lambda) with Pw from Erlang-C.

    Inputs:
      lambda_ : arrival rate (Poisson)
      mu      : effective service rate per server (mu = 1/E[S])  [derived for general service]
      c       : number of servers
      varS    : variance of service time
    """
    if lambda_ <= 0:
        raise ValueError("lambda must be > 0")
    if mu <= 0:
        raise ValueError("mu must be > 0")
    if c <= 0:
        raise ValueError("servers c must be >= 1")
    if varS < 0:
        raise ValueError("varS must be >= 0")

    ES = 1.0 / mu
    varA = 1.0 / (lambda_ * lambda_)  # Poisson arrivals => exponential interarrivals

    # Utilization
    rho = lambda_ / (c * mu)

    # Unstable check
    if rho >= 1.0:
        return AnalyticalResult(
            interarrival_rate=r4(lambda_),
            service_rate=r4(mu),
            utilization=r4(rho),
            var_services=r4(varS),
            var_interarrivals=r4(varA),
            Lq=float("inf"),
            Wq=float("inf"),
            W=float("inf"),
            L=float("inf"),
            note="Unstable system (λ ≥ cμ)"
        )

    # Erlang-C waiting probability as M/M/c baseline
    Pw, _ = _erlang_c(lambda_, mu, c)

    # Base M/M/c waiting time
    Wq_mmc = Pw / (c * mu - lambda_)

    # Service SCV Cs^2 = Var(S)/(E[S]^2)
    Cs2 = _scv(varS, ES)

    # Allen–Cunneen factor
    factor = (1.0 + Cs2) / 2.0

    # Approx waiting time for M/G/c
    Wq = factor * Wq_mmc
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
def solve_mgc_from_service_choice(
    lambda_: float,
    c: int,
    service_type: str,
    service_params: Dict[str, float],
) -> AnalyticalResult:
    """
    Backend wrapper for M/G/c:
      - Arrivals are Poisson: only lambda_ needed for arrivals
      - Service is chosen: uniform/gamma/normal
      - Converts service params -> mu_eff (=1/E[S]) and varS
      - Calls mgc(lambda_, mu_eff, c, varS)
      - Adds a practical warning for Normal if std is large vs mean
    """
    if lambda_ <= 0:
        raise ValueError("lambda must be > 0")
    if c <= 0:
        raise ValueError("servers c must be >= 1")
    if not service_type:
        raise ValueError("service_type is required")
    if service_params is None:
        raise ValueError("service_params is required")

    st = service_type.strip().lower()
    warning_note = None

    if st == "uniform":
        a = float(service_params["min"])
        b = float(service_params["max"])
        if b <= a:
            raise ValueError("uniform requires max > min")
        meanS = (a + b) / 2.0
        varS = ((b - a) ** 2) / 12.0

    elif st == "gamma":
        k = float(service_params["shape"])
        scale = float(service_params["scale"])
        if k <= 0 or scale <= 0:
            raise ValueError("gamma requires shape > 0 and scale > 0")
        meanS = k * scale
        varS = k * (scale ** 2)


    elif st == "normal":
        meanS = float(service_params["mean"])
        std = float(service_params["std"])
        if meanS <= 0:
            raise ValueError("normal requires mean > 0 for service times")
        if std < 0:
            raise ValueError("normal requires std >= 0")
        varS = std ** 2

        # Practical modeling warning (not a bug)
        if std > meanS / 2:
            warning_note = (
                "Normal service may produce negative service times; "
                "consider using a gamma distribution."
            )
    else:
        raise ValueError('service_type must be one of: "uniform", "gamma", "normal"')

    mu_eff = 1.0 / meanS
    result = mgc(lambda_=lambda_, mu=mu_eff, c=c, varS=varS)

    if warning_note:
        result.note = warning_note

    return result

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
# def solve_analytical(model: str, lambda_: float, mu: float, c: int,
#                      arrival_spec: Optional[Dict] = None,
#                      service_spec: Optional[Dict] = None) -> AnalyticalResult:
#     """
#     Supports ALL:
#       - M/M/1
#       - M/M/c
#       - M/G/1
#       - M/G/c
#       - G/G/1
#       - G/G/c
#
#     G distributions can be: normal, uniform, gamma
#     """
#
#     # Normalize model string
#     m = model.strip().upper().replace(" ", "")
#
#     # For G/G models, arrival & service distributions are mandatory
#     # because UI does NOT provide lambda/mu
#     if m in ["G/G/1", "GG1", "G/G/C", "GGC", "GG/C"]:
#         if arrival_spec is None or service_spec is None:
#             raise ValueError(
#                 "For G/G models, arrival_spec and service_spec are required"
#             )
#
#     # Default arrival only for NON-GG models
#     if arrival_spec is None and m not in ["G/G/1", "GG1", "G/G/C", "GGC", "GG/C"]:
#         arrival_spec = {"dist_type": "exponential", "params": {"rate": lambda_}}
#
#     # Only default service_spec if the model is NOT M/G/1 or M/G/C
#     if service_spec is None and m not in ["M/G/1", "MG1", "M/G/C", "MGC", "MG/C"]:
#         service_spec = {"dist_type": "exponential", "params": {"rate": mu}}
#
#     # Compute arrival mean & variance (always safe)
#     meanA, varA = mean_variance_from_spec(arrival_spec)
#
#     # Compute service mean & variance only if we have a service_spec
#     meanS = varS = None
#     if service_spec is not None:
#         meanS, varS = mean_variance_from_spec(service_spec)
#
#     # ---- Exact models ----
#     if m in ["M/M/1", "MM1"]:
#         return mm1(lambda_, mu)
#
#     if m in ["M/M/C", "MMC", "MM/C"]:
#         return mmc(lambda_, mu, c)
#
#     # ---- Mixed / General models ----
#     if m in ["M/G/1", "MG1"]:
#         # Arrivals fixed as Poisson (M). Service distribution is chosen by service_spec.
#         if service_spec is None:
#             raise ValueError("service_spec is required for M/G/1 when using distribution parameters")
#
#         dist = service_spec.get("dist_type", "").lower()
#         p = service_spec.get("params", {}) or {}
#
#         if dist == "uniform":
#             return mg1(lambda_, "uniform", {"min": p.get("min"), "max": p.get("max")})
#         elif dist == "normal":
#             return mg1(lambda_, "normal", {"mean": p.get("mean"), "std": p.get("std")})
#         elif dist == "gamma":
#             return mg1(lambda_, "gamma", {"k": p.get("shape") or p.get("k"), "theta": p.get("scale") or p.get("theta")})
#         else:
#             raise ValueError("For M/G/1, service_spec.dist_type must be: uniform, normal, or gamma")
#
#     if m in ["M/G/C", "MGC", "MG/C"]:
#         if service_spec is None:
#             raise ValueError("service_spec is required for M/G/c")
#
#         dist = service_spec.get("dist_type", "").lower()
#         params = service_spec.get("params", {}) or {}
#
#         return solve_mgc_from_service_choice(
#             lambda_=lambda_,
#             c=c,
#             service_type=dist,
#             service_params=params
#         )
#
#     # if m in ["G/G/1", "GG1"]:
#     #     return gg1(lambda_, mu, varA, varS)
#     if m in ["G/G/1", "GG1"]:
#         if arrival_spec is None or service_spec is None:
#             raise ValueError("arrival_spec and service_spec are required for G/G/1")
#
#         meanA, varA = mean_variance_from_spec(arrival_spec)
#         meanS, varS = mean_variance_from_spec(service_spec)
#
#         if meanA <= 0 or meanS <= 0:
#             raise ValueError("Computed mean times must be > 0")
#
#         lambda_eff = 1.0 / meanA
#         mu_eff = 1.0 / meanS
#
#         return gg1(lambda_eff, mu_eff, varA, varS)
#
#     # if m in ["G/G/C", "GGC", "GG/C"]:
#     #     return ggc(lambda_, mu, c, varA, varS)
#     if m in ["G/G/C", "GGC", "GG/C"]:
#         if arrival_spec is None or service_spec is None:
#             raise ValueError("arrival_spec and service_spec are required for G/G/c")
#
#         meanA, varA = mean_variance_from_spec(arrival_spec)
#         meanS, varS = mean_variance_from_spec(service_spec)
#
#         if meanA <= 0 or meanS <= 0:
#             raise ValueError("Computed mean times must be > 0")
#
#         lambda_eff = 1.0 / meanA
#         mu_eff = 1.0 / meanS
#
#         return ggc(lambda_eff, mu_eff, c, varA, varS)
#
#     raise NotImplementedError(f"Model not implemented yet: {model}")
def solve_analytical(model: str,
                     lambda_: Optional[float],
                     mu: Optional[float],
                     c: int,
                     arrival_spec: Optional[Dict] = None,
                     service_spec: Optional[Dict] = None) -> AnalyticalResult:

    m = model.strip().upper().replace(" ", "")

    gg_models = ["G/G/1", "GG1", "G/G/C", "GGC", "GG/C"]

    # GG requires both specs (no lambda/mu input)
    if m in gg_models:
        if arrival_spec is None or service_spec is None:
            raise ValueError("For G/G models, arrival_spec and service_spec are required")

    # Defaults ONLY for non-GG models
    if arrival_spec is None:
        if lambda_ is None or lambda_ <= 0:
            raise ValueError("lambda must be provided (>0) for non-GG models")
        arrival_spec = {"dist_type": "exponential", "params": {"rate": lambda_}}

    if service_spec is None and m not in ["M/G/1", "MG1", "M/G/C", "MGC", "MG/C"]:
        if mu is None or mu <= 0:
            raise ValueError("mu must be provided (>0) for models that default service to exponential")
        service_spec = {"dist_type": "exponential", "params": {"rate": mu}}

    # compute moments once
    meanA, varA = mean_variance_from_spec(arrival_spec)
    meanS = varS = None
    if service_spec is not None:
        meanS, varS = mean_variance_from_spec(service_spec)

    # ---- Exact models ----
    if m in ["M/M/1", "MM1"]:
        if lambda_ is None or mu is None:
            raise ValueError("lambda and mu are required for M/M/1")
        return mm1(lambda_, mu)

    if m in ["M/M/C", "MMC", "MM/C"]:
        if lambda_ is None or mu is None:
            raise ValueError("lambda and mu are required for M/M/c")
        return mmc(lambda_, mu, c)

    # ---- M/G/1 ----
    if m in ["M/G/1", "MG1"]:
        if lambda_ is None or lambda_ <= 0:
            raise ValueError("lambda is required for M/G/1")
        if service_spec is None:
            raise ValueError("service_spec is required for M/G/1")
        dist = service_spec.get("dist_type", "").lower()
        p = service_spec.get("params", {}) or {}

        if dist == "uniform":
            return mg1(lambda_, "uniform", {"min": p.get("min"), "max": p.get("max")})
        elif dist == "normal":
            return mg1(lambda_, "normal", {"mean": p.get("mean"), "std": p.get("std")})
        elif dist == "gamma":
            return mg1(lambda_, "gamma", {"shape": p.get("shape"), "scale": p.get("scale")})
        else:
            raise ValueError("For M/G/1, service_spec.dist_type must be: uniform, normal, gamma")

    # ---- M/G/c ----
    if m in ["M/G/C", "MGC", "MG/C"]:
        if lambda_ is None or lambda_ <= 0:
            raise ValueError("lambda is required for M/G/c")
        if service_spec is None:
            raise ValueError("service_spec is required for M/G/c")
        dist = service_spec.get("dist_type", "").lower()
        params = service_spec.get("params", {}) or {}
        return solve_mgc_from_service_choice(lambda_=lambda_, c=c, service_type=dist, service_params=params)

    # ---- G/G/1 ----
    if m in ["G/G/1", "GG1"]:
        if meanA <= 0 or meanS is None or meanS <= 0:
            raise ValueError("Computed mean times must be > 0")
        lambda_eff = 1.0 / meanA
        mu_eff = 1.0 / meanS
        return gg1(lambda_eff, mu_eff, varA, varS)

    # ---- G/G/c ----
    if m in ["G/G/C", "GGC", "GG/C"]:
        if meanA <= 0 or meanS is None or meanS <= 0:
            raise ValueError("Computed mean times must be > 0")
        lambda_eff = 1.0 / meanA
        mu_eff = 1.0 / meanS
        return ggc(lambda_eff, mu_eff, c, varA, varS)

    raise NotImplementedError(f"Model not implemented yet: {model}")
