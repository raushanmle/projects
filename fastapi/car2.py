from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import json
import uvicorn

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
        "id": "1s"
    },
]


class carinput(BaseModel):
    size: str
    doors: int
    fuel: str
    transmission: str
    trips: list
    id: int


def load_db():
    result = [carinput.model_validate(ip) for ip in json.load(open("cars1.json"))]
    return result


def save_json(result):
    json.dump([res.model_dump() for res in result], open("cars1.json", 'w'), indent= 4)

app = FastAPI(title = " Car Details")
result = load_db()
@app.get("/{id}")

def get_cars(id: int):
    for cr in result:
        if cr.id == id:
            return cr.model_dump()
        
    raise HTTPException(status_code=200, detail=f"No car with id={id}.")



if __name__ == "__main__":
    uvicorn.run("car2:app", reload= True)

