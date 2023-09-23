from flask import Flask, jsonify, request

app = Flask(__name__)

books_db = [

    {
        "name": "Billa",
        "write": "Bilar"

    },
    {
        "name": "Billa 2",
        "write": "Bilar 2"
    }
]


@app.route("/")
def home():
    return "Hello Babe1"


@app.route("/books")
def get_all_books():
    return jsonify({"books": books_db})

@app.route("/book/<string:name>")

def get_book(name):
    for book in books_db:
        if book["name"] == name:
            return jsonify(book)
        else:
            return jsonify({"message": "book not found"})


@app.route("/book", methods = ['POST'])

def create_entry():
    body_data = request.get_json()
    books_db.append(body_data)
    return jsonify({"message": "entry added successfully"})


app.run(port=5001)