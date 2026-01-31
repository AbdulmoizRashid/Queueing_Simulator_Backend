from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AnalyticalRequest, AnalyticalResponse,
    SimulationRequest, SimulationResponse
)

from core.analytical import solve_analytical
from core.simulation import simulate
from core.models import DistributionSpec as CoreDistributionSpec, SimulationRequest as CoreSimReq

app = FastAPI(title="Queue Simulator API", version="1.0")

# allow frontend (React/etc.) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _to_core_dist(d):
    # convert pydantic schema -> core dataclass
    return CoreDistributionSpec(dist_type=d.dist_type, params=d.params)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analytical", response_model=AnalyticalResponse)
def analytical(req: AnalyticalRequest):
    # If arrival/service not provided, solver will default to exponential using lambda/mu
    arrival_spec = req.arrival.dict() if req.arrival else None
    service_spec = req.service.dict() if req.service else None

    res = solve_analytical(
        model=req.model,
        lambda_=req.lambda_,
        mu=req.mu,
        c=req.servers,
        arrival_spec=arrival_spec,
        service_spec=service_spec
    )
    return AnalyticalResponse(**res.__dict__)

@app.post("/simulate", response_model=SimulationResponse)
def simulate_endpoint(req: SimulationRequest):
    core_req = CoreSimReq(
        model=req.model,
        servers=req.servers,
        n_customers=req.n_customers,
        arrival=_to_core_dist(req.arrival),
        service=_to_core_dist(req.service),
        seed=req.seed
    )
    res = simulate(core_req)

    # convert dataclasses -> dicts for pydantic response
    return SimulationResponse(
        rows=[r.__dict__ for r in res.rows],
        gantt=[g.__dict__ for g in res.gantt],
        wait_times=res.wait_times,
        turnaround_times=res.turnaround_times,
        response_times=res.response_times,
        arrival_times=res.arrival_times
    )
