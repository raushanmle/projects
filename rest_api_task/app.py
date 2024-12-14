from flask import Flask, jsonify, request

app = Flask(__name__)
app.config['DEBUG'] = True



@app.get('/store')
def get_stores():
    return jsonify({'stores': stores})


@app.post('/store')
def create_store():
    request_data = request.get_json()
    new_store = {
        'name': request_data['name'],
        'items': []
    }
    stores.append(new_store)
    return jsonify(new_store)


stores = [
    {
        'name': 'mystore1',
        'items': [
            {
                'name': 'item1',
                'price': 15.99
            }
        ]
    }
]


@app.post('/store/<string:name>')
def create_item_in_store(name):
    print(name)
    request_data = request.get_json()
    for store in stores:
        if store['name'] == name:
            new_item = {
                'name': request_data['name'],
                'price': request_data['price']
            }
            store['items'].append(new_item)
            return jsonify(new_item), 201
    return jsonify({'message': 'store not found'}), 404

if __name__ == '__main__':
    app.run()




