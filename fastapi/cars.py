from pydantic import BaseModel

class carinput(BaseModel):
    size: str
    fuel: str
    doors: int
    transmission: str
    trips: list
    id: int

class TripInput(BaseModel):
    start: int
    end: int
    description: str
    id: int

#res = carinput.model_validate(cr[0])



cr = [
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
]

import json
def load_db():
    """Load a list of Car objects from a JSON file"""
    with open("cars.json") as f:
        return json.load(f)


def save_db(cars):
    with open("cars.json", 'w') as f:
        json.dump(cars, f, indent=4)

import uvicorn
from fastapi import FastAPI, HTTPException
app = FastAPI(title="Car Sharing")
@app.get("/api/cars/{id}")
def car_by_id(id: int) -> dict:
    result = [car for car in cr if car["id"] == id]
    if result:
        return dict(result[0])
    else:
        raise HTTPException(status_code=404, detail=f"No car with id={id}.")

db = load_db()

@app.delete("/api/cars/{id}", status_code=204)
def remove_car(id: int) -> None:
    matches = [car for car in db if car["id"] == id]
    if matches:
        car = matches[0]
        db.remove(car)
        save_db(db)
    else:
        raise HTTPException(status_code=404, detail=f"No car with id={id}.")

@app.put("/api/cars/{id}")
def change_car(id: int, new_data:dict):
    matches = [car for car in db if car["id"] == id]
    if matches:
        car = matches[0]
        car["fuel"] = new_data["fuel"]
        car["transmission"] = new_data["transmission"]
        car["size"] = new_data["size"]
        car["doors"] = new_data["doors"]
        save_db(db)
        return car
    else:
        raise HTTPException(status_code=404, detail=f"No car with id={id}.")


if __name__ == "__main__":
    # uvicorn.run("carsharing:app", port = 8000, host = "0.0.0.0", reload= True)
    uvicorn.run("cars:app", reload= True)