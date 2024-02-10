from pydantic import BaseModel
from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException
import json

# Definning input type
class TripInput(BaseModel):
    start: int
    end: int
    description: str


class TripOutput(TripInput):
    id: int

class CarInput(BaseModel):
    size: str
    fuel: str | None = "electric"
    doors: int
    transmission: str | None = "auto"

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

def load_db():
    """Load a list of Car objects from a JSON file"""
    with open("cars.json") as f:
        return json.load(f)
db = load_db()

def save_db(cars):
    with open("cars.json", 'w') as f:
        json.dump(cars, f, indent=4)


app = FastAPI(title="Car Sharing API Testing")

@app.get("/api/cars/{id}")
def car_by_id(id: int) -> dict:
    result = [car for car in db if car["id"] == id]
    if result:
        return dict(result[0])
    else:
        raise HTTPException(status_code=404, detail=f"No car with id={id}.")


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

# @app.post("/api/cars/")
# def add_car(car: dict):
#     new_car = {"size": car["size"], "doors":car["doors"],"fuel":car["fuel"], "transmission":car["transmission"],"trips":[], "id":len(db)+1}

#     db.append(new_car)
#     save_db(db)
#     return {"Entry added": new_car}


@app.post("/api/cars/")
def add_car(car: CarInput):
    new_car = car.dict()
    new_car["id"] = len(db) + 1
    db.append(new_car)
    save_db(db)
    return {"Entry added": new_car}



from sqlmodel import create_engine, SQLModel, Session, select
engine = create_engine(
    "sqlite:///carsharing.db",
    connect_args={"check_same_thread": False},
    echo=True
)

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session



if __name__ == "__main__":
    # uvicorn.run("carsharing:app", port = 8000, host = "0.0.0.0", reload= True)
    uvicorn.run("cars:app", reload= True)