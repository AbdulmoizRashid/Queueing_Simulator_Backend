from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Literal

DistType = Literal["exponential", "uniform", "normal", "gamma"]

class DistributionSpec(BaseModel):
    dist_type: DistType
    params: Dict[str, float]

# ---------- Analytical ----------
class AnalyticalRequest(BaseModel):
    model: str = Field(..., examples=["M/M/1", "M/M/c", "M/G/1", "G/G/1", "G/G/c"])
    lambda_: float = Field(..., gt=0)
    mu: float = Field(..., gt=0)
    servers: int = Field(1, ge=1)

    # only needed when model includes G
    arrival: Optional[DistributionSpec] = None
    service: Optional[DistributionSpec] = None

class AnalyticalResponse(BaseModel):
    interarrival_rate: float
    service_rate: float
    utilization: float
    var_services: float
    var_interarrivals: float
    Lq: float
    Wq: float
    W: float
    L: float
    note: Optional[str] = None

# ---------- Simulation ----------
class SimulationRequest(BaseModel):
    model: str = Field(..., examples=["M/M/c", "G/G/c"])
    servers: int = Field(..., ge=1)
    n_customers: int = Field(..., ge=1, le=200000)
    arrival: DistributionSpec
    service: DistributionSpec
    seed: Optional[int] = 123

class SimulationRow(BaseModel):
    serial_no: int
    cp: float
    cp_lookup: str
    avg_time_between_arrivals: float
    inter_arrival_time: float
    arrival_time: float
    service_start_time: float
    service_end_time: float
    service_time: float
    waiting_time: float
    turnaround_time: float
    response_time: float
    server_time: str

class GanttBlock(BaseModel):
    server_id: int
    customer_id: int
    start: float
    end: float

class SimulationResponse(BaseModel):
    rows: List[SimulationRow]
    gantt: List[GanttBlock]
    wait_times: List[float]
    turnaround_times: List[float]
    response_times: List[float]
    arrival_times: List[float]
