import uvicorn
from fastapi import FastAPI, HTTPException

from schemas import load_db, save_db, CarInput, CarOutput, TripOutput, TripInput

app = FastAPI(title="Car Sharing")
db = [
    {
        "size": "s",
        "fuel": "gasoline",
        "doors": 3,
        "transmission": "auto",
        "trips": [
            {
                "start": 0,
                "end": 5,
                "description": "Groceries",
                "id": 1
            },
            {
                "start": 5,
                "end": 218,
                "description": "Commute Amsterdam-Rotterdam",
                "id": 2
            },
            {
                "start": 218,
                "end": 257,
                "description": "Weekend beach trip",
                "id": 3
            }
        ],
        "id": 1
    },
    {
        "size": "s",
        "fuel": "electric",
        "doors": 3,
        "transmission": "auto",
        "trips": [
            {
                "start": 0,
                "end": 34,
                "description": "Taking dog to the vet",
                "id": 4
            },
            {
                "start": 34,
                "end": 125,
                "description": "Meeting Customer in Utrecht",
                "id": 5
            }
        ],
        "id": 2
    },
    {
        "size": "s",
        "fuel": "gasoline",
        "doors": 5,
        "transmission": "manual",
        "trips": [],
        "id": 3
    },
    {
        "size": "m",
        "fuel": "electric",
        "doors": 3,
        "transmission": "auto",
        "trips": [
            {
                "start": 0,
                "end": 100,
                "description": "Visiting mom",
                "id": 6
            }
        ],
        "id": 4
    },
    {
        "size": "m",
        "fuel": "gasoline",
        "doors": 5,
        "transmission": "manual",
        "trips": [],
        "id": 6
    },
    {
        "size": "l",
        "fuel": "diesel",
        "doors": 5,
        "transmission": "manual",
        "trips": [],
        "id": 7
    },
    {
        "size": "l",
        "fuel": "electric",
        "doors": 5,
        "transmission": "auto",
        "trips": [],
        "id": 8
    },
    {
        "size": "l",
        "fuel": "hybrid",
        "doors": 5,
        "transmission": "auto",
        "trips": [
            {
                "start": 0,
                "end": 55,
                "description": "Forest walk",
                "id": 7
            }
        ],
        "id": 9
    },
    {
        "size": "xl",
        "fuel": "electric",
        "doors": 5,
        "transmission": "auto",
        "trips": [],
        "id": 10
    }
]


@app.get("/api/cars")
def get_cars(size: str|None = None, doors: int|None = None) -> list:
    if size:
        return [car for car in db if car["size"] == size]
    elif doors:
        return [car for car in db if car["doors"] == doors]
    else:
        return db

@app.get("/api/cars/{id}")
def car_by_id(id: int):
    res = [car for car in db if car["id"] == id]
    if res:
        return res[0]
    else:
        raise HTTPException(status_code=404, detail= f"No car found with id : {id}")

