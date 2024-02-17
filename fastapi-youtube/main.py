from fastapi import FastAPI
import uvicorn
from enum import Enum

app = FastAPI(title="FAST API")


@app.get("/")

async def root():
    return {"message": "hello: what's up"}

class FoodEnum(str, Enum):
    mango = "mango"
    banana = "banana"
    apple = "apple"

@app.get("/{fruit_name}")
async def show_food(fruit_name:FoodEnum):
    if fruit_name.value == FoodEnum.mango:
        return {"message": "you are healthy"}
    elif fruit_name.value == FoodEnum.apple:
        return {"message": "you have selected apple"}
    

from pydantic import BaseModel
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float

@app.post("/items")
async def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax" : price_with_tax})
    return item_dict


@app.put("/items/{item_id}")
async def create_item_id_with_pt(item_id: int, item: Item, q: str | None = None):
    res = {"item_id": item_id, **item.dict()}
    if q:
        res.update({"q": q})
    return res




if __name__ == "__main__":
    uvicorn.run("main:app", reload= True)