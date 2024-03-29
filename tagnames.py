import requests
import json
import os

#this script creates a locally stored list of tags 
#To know which TAGS might work in the Model tag search and which do not. 
#Because there is also stuff on the list from the API that are not working. e.g. comma seperated words 
#adds only new tags that are not yet present in the tag_names.txt file


def fetch_data(url):
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"Invalid JSON-data: {str(e)}")
        return response.json()
    else:
        print(f"Error while retrieving data, Status-Code: {response.status_code}")
    return None


def process_data(items, file_path, read_existing_tag):
    with open(file_path, 'a') as file:
        for item in items:
            name =  item.get('name')
            if name and name not in existing_tags:
                file.write(name+ '\n')
                existing_tags.add(name)


def read_existing_tag(file_path):
    try:
        with open(file_path, 'r') as file:
            return {line.strip() for line in file}
    except FileNotFoundError:
        return set()



file_path = os.path.join(os.getcwd(), 'tag_names.txt')
existing_tags = read_existing_tag(file_path)
url = "https://civitai.com/api/v1/tags?limit=200"

while url:
    data = fetch_data(url)
    if data:
        items = data.get('items', [])
        process_data(items, file_path, read_existing_tag)
        existing_tags.update(item.get('name') for item in items if item.get('name')) 

        metadata = data.get('metadata', {})
        url = metadata.get('nextPage')
    else:
        url = None
