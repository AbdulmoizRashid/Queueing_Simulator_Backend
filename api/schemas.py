from typing import Dict, Optional, List, Literal
from pydantic import BaseModel, Field, model_validator


DistType = Literal["exponential", "uniform", "normal", "gamma"]

class DistributionSpec(BaseModel):
    dist_type: DistType
    params: Dict[str, float]

# ---------- Analytical ----------
class AnalyticalRequest(BaseModel):
    model: str = Field(
        ...,
        examples=["M/M/1", "M/M/c", "M/G/1", "M/G/c", "G/G/1", "G/G/c"]
    )

    lambda_: Optional[float] = Field(None, gt=0)
    mu: Optional[float] = Field(None, gt=0)
    servers: int = Field(1, ge=1)

    arrival: Optional[DistributionSpec] = None
    service: Optional[DistributionSpec] = None

    @model_validator(mode="after")
    def check_model_requirements(self):
        model = (self.model or "").strip().upper().replace(" ", "")
        lam = self.lambda_
        mu = self.mu
        arrival = self.arrival
        service = self.service

        # M/M/*
        if model in ["M/M/1", "MM1", "M/M/C", "MMC", "MM/C"]:
            if lam is None or mu is None:
                raise ValueError("M/M models require lambda_ and mu")
            return self

        # M/G/*
        if model in ["M/G/1", "MG1", "M/G/C", "MGC", "MG/C"]:
            if lam is None:
                raise ValueError("M/G models require lambda_")
            if service is None:
                raise ValueError("M/G models require service distribution")
            return self

        # G/G/*
        if model in ["G/G/1", "GG1", "G/G/C", "GGC", "GG/C"]:
            if arrival is None or service is None:
                raise ValueError("G/G models require arrival and service distributions")
            return self

        raise ValueError(f"Unknown model: {model}")


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
