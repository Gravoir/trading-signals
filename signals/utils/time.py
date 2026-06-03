"""Time utilities."""
from datetime import datetime, timezone

def timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds."""
    multipliers = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400, 'w': 604800}
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    return value * multipliers.get(unit, 60)

def align_to_timeframe(timestamp: int, timeframe: str) -> int:
    """Align timestamp to timeframe boundary."""
    seconds = timeframe_to_seconds(timeframe)
    return (timestamp // seconds) * seconds
