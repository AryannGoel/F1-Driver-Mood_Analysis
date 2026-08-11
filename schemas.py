"""Output contract shared with the Pit Wall frontend.

The `laps` array maps 1:1 to the `STINT` array in PitWall.jsx, so the UI
drops in unchanged whether the data came from demo mode or the live pipeline.
"""
from typing import Optional, List, Literal
from pydantic import BaseModel

Mood = Literal["calm", "focused", "stressed", "tired"]


class Lap(BaseModel):
    lap: int
    t: float                      # lap time in seconds
    stress: int                   # 0-100, derived from the emotion model
    mood: Mood
    radio: Optional[str] = None   # transcript, or null on laps with no radio
    tone: Optional[str] = None    # raw detected emotion of the driver's voice
    confidence: Optional[int] = None  # 0-100, model confidence in that emotion


class Meta(BaseModel):
    driver: str
    team: str
    session: str
    compound: str = "HARD"
    stint: str
    source: str                   # "live"


class Stint(BaseModel):
    meta: Meta
    laps: List[Lap]
