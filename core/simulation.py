import random
from typing import List
from .models import SimulationRequest, SimulationResult, SimulationRow, GanttBlock
from .distributions import sample_from_spec

def simulate(req: SimulationRequest) -> SimulationResult:
    rng = random.Random(req.seed)

    c = req.servers
    N = req.n_customers

    server_available = [0.0] * c
    rows: List[SimulationRow] = []
    gantt: List[GanttBlock] = []

    arrival_time = 0.0
    interarrival_sum = 0.0

    wait_times = []
    turnaround_times = []
    response_times = []
    arrival_times = []

    for i in range(1, N + 1):
        inter_arrival, cp_a, lookup_a = sample_from_spec(req.arrival.__dict__, rng)
        service_time, cp_s, lookup_s = sample_from_spec(req.service.__dict__, rng)

        arrival_time += inter_arrival
        arrival_times.append(arrival_time)

        interarrival_sum += inter_arrival
        avg_between = interarrival_sum / i

        # pick earliest available server
        server_idx = min(range(c), key=lambda k: server_available[k])

        start = max(arrival_time, server_available[server_idx])
        end = start + service_time

        waiting = start - arrival_time
        turnaround = end - arrival_time
        response = waiting + service_time

        server_available[server_idx] = end

        gantt.append(GanttBlock(
            server_id=server_idx + 1,
            customer_id=i,
            start=start,
            end=end
        ))

        # CP + CP Lookup: show arrival CP as "CP" and service lookup in CP Lookup (you can change)
        # You can also store cp_a and cp_s separately later.
        row = SimulationRow(
            serial_no=i,
            cp=cp_a,
            cp_lookup=f"A:{lookup_a} | S:{lookup_s}",
            avg_time_between_arrivals=avg_between,
            inter_arrival_time=inter_arrival,
            arrival_time=arrival_time,
            service_start_time=start,
            service_end_time=end,
            service_time=service_time,
            waiting_time=waiting,
            turnaround_time=turnaround,
            response_time=response,
            server_time=f"Server {server_idx + 1}"
        )
        rows.append(row)

        wait_times.append(waiting)
        turnaround_times.append(turnaround)
        response_times.append(response)

    return SimulationResult(
        rows=rows,
        gantt=gantt,
        wait_times=wait_times,
        turnaround_times=turnaround_times,
        response_times=response_times,
        arrival_times=arrival_times
    )
