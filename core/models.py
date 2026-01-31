from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Any

DistType = Literal["exponential", "uniform", "normal", "gamma"]

@dataclass
class DistributionSpec:
    dist_type: DistType
    params: Dict[str, float]  # e.g. {"rate": 2.0} or {"min":1,"max":3}

@dataclass
class SimulationRequest:
    model: str                 # e.g. "M/M/c", "G/G/c"
    servers: int               # c
    n_customers: int           # N
    arrival: DistributionSpec  # inter-arrival distribution
    service: DistributionSpec  # service-time distribution
    seed: Optional[int] = 123  # reproducible by default

@dataclass
class SimulationRow:
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
    server_time: str  # store "Server 1", "Server 2" etc.

@dataclass
class GanttBlock:
    server_id: int
    customer_id: int
    start: float
    end: float

@dataclass
class SimulationResult:
    rows: List[SimulationRow]
    gantt: List[GanttBlock]
    wait_times: List[float]
    turnaround_times: List[float]
    response_times: List[float]
    arrival_times: List[float]
