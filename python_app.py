import json

def open_json_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

print(open_json_file('functions.json'))