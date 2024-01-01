import json

from pydantic import BaseModel
from typing import Optional


class TripInput(BaseModel):
    start: int
    end: int
    description: str


class TripOutput(TripInput):
    id: int


class CarInput(BaseModel):
    size: str
    fuel: Optional[str] = "electric"
    doors: int
    transmission: Optional[str] = "auto"

    class Config:
        json_schema_extra = {
            "example": {
                "size": "m",
                "doors": 5,
                "transmission": "manual",
                "fuel": "hybrid"
            }
        }


class CarOutput(CarInput):
    id: int
    trips: list[TripOutput] = []


def load_db() -> list[CarOutput]:
    """Load a list of Car objects from a JSON file"""
    with open("C:\\Code\\projects\\fastapi\\cars.json") as f:
        return [CarOutput.BaseModel(obj) for obj in json.load(f)]


def save_db(cars: list[CarOutput]):
    with open("cars.json", 'w') as f:
        json.dump([car.BaseModel() for car in cars], f, indent=4)
