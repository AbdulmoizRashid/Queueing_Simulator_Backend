# run.py

from core.models import SimulationRequest, DistributionSpec
from core.simulation import simulate
from core.analytical import solve_analytical

# =====================================================
# 1️⃣ Simulation
# =====================================================
req = SimulationRequest(
    model="M/M/c",
    servers=2,
    n_customers=10,
    arrival=DistributionSpec(
        dist_type="exponential",
        params={"rate": 0.6}
    ),
    service=DistributionSpec(
        dist_type="exponential",
        params={"rate": 1.0}
    ),
    seed=42
)

res = simulate(req)

print("=== Simulation (first 5 customers) ===")
for r in res.rows[:5]:
    print(r)

print("Gantt blocks:", res.gantt[:5])

# =====================================================
# 2️⃣ Analytical — M/M models
# =====================================================
print("\n=== Analytical M/M/1 ===")
res_mm1 = solve_analytical(
    model="M/M/1",
    lambda_=0.6,
    mu=1.0,
    c=1
)
print(res_mm1)

print("\n=== Analytical M/M/c ===")
res_mmc = solve_analytical(
    model="M/M/c",
    lambda_=0.8,
    mu=1.0,
    c=2
)
print(res_mmc)

# =====================================================
# 3️⃣ Analytical — General models
# =====================================================
print("\n=== Analytical M/G/1 (service gamma) ===")
mg1_res = solve_analytical(
    model="M/G/1",
    lambda_=0.6,
    mu=None,
    c=1,
    service_spec={
        "dist_type": "gamma",
        "params": {"shape": 2.0, "scale": 0.5}
    }
)

print(mg1_res)

print("\n=== Analytical M/G/c test ===")
res = solve_analytical(
    model="M/G/c",
    lambda_=1.6,
    mu=None,
    c=2,
    service_spec={
        "dist_type": "gamma",
        "params": {"shape": 2.0, "scale": 0.5}
    }
)

print(res)

print("\n=== Analytical G/G/1 (uniform arrivals, normal service) ===")
gg1_res = solve_analytical(
    model="G/G/1",
    lambda_=None,
    mu=None,
    c=1,
    arrival_spec={
        "dist_type": "uniform",
        "params": {"min": 0.5, "max": 2.5}
    },
    service_spec={
        "dist_type": "normal",
        "params": {"mean": 1.0, "std": 0.2}
    }
)

print(gg1_res)

print("\n=== Analytical G/G/c (gamma arrivals, gamma service, c=2) ===")
ggc_res = solve_analytical(
    model="G/G/c",
    lambda_=None,
    mu=None,
    c=2,
    arrival_spec={
        "dist_type": "gamma",
        "params": {"shape": 2.0, "scale": 1.0}
    },
    service_spec={
        "dist_type": "gamma",
        "params": {"shape": 2.0, "scale": 0.5}
    }
)

print(ggc_res)
