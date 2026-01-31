def require_positive(name: str, value: float) -> None:
    if value is None or value <= 0:
        raise ValueError(f"{name} must be > 0")

def require_non_negative(name: str, value: float) -> None:
    if value is None or value < 0:
        raise ValueError(f"{name} must be >= 0")

def require_int_at_least(name: str, value: int, minimum: int) -> None:
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
