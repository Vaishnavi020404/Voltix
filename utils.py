import face_recognition as frg
import pickle as pkl 
import os 
import cv2 
import numpy as np
import yaml
from collections import defaultdict

information = defaultdict(dict)
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
DATASET_DIR = cfg['PATH']['DATASET_DIR']
PKL_PATH = cfg['PATH']['PKL_PATH']

def get_databse():
    with open(PKL_PATH,'rb') as f:
        database = pkl.load(f)
    return database
def recognize(image, TOLERANCE):
    database = get_databse()
    known_encoding = [database[idx]['encoding'] for idx in database.keys()]
    name = 'Unknown'
    id = 'Unknown'
    face_locations = frg.face_locations(image)
    face_encodings = frg.face_encodings(image, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = frg.compare_faces(known_encoding, face_encoding, tolerance=TOLERANCE)
        distances = frg.face_distance(known_encoding, face_encoding)
        name = 'Unknown'
        id = 'Unknown'
        if True in matches:
            match_index = matches.index(True)
            name = database[match_index]['name']
            id = database[match_index]['id']
            _ = round(distances[match_index], 2)  # computed but not displayed
        # Draw face box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Label: show MoodleID_Name (underscored) when known; otherwise 'Unknown'
        label = 'Unknown' if id == 'Unknown' else f"{id}_{name.replace(' ', '_')}"
        # Position label
        y_label = top - 10
        if y_label < 20:
            y_label = top + 20
        # Background box for readability
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        box_top_left = (left, y_label - text_h - baseline - 4)
        box_bottom_right = (left + text_w + 8, y_label + 4)
        cv2.rectangle(image, box_top_left, box_bottom_right, (0, 255, 0), thickness=-1)
        cv2.putText(image, label, (left + 4, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image, name, id
def isFaceExists(image):
    face_location = frg.face_locations(image)
    if len(face_location) == 0:
        return False
    return True
def submitNew(name, id, image, old_idx=None):
    database = get_databse()
    # Read image
    if type(image) != np.ndarray:
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)

    isFaceInPic = isFaceExists(image)
    if not isFaceInPic:
        return -1
    # Encode image
    encoding = frg.face_encodings(image)[0]
    # Append to database
    existing_id = [database[i]['id'] for i in database.keys()]
    # Update mode
    if old_idx is not None:
        new_idx = old_idx
    else:
        if id in existing_id:
            return 0
        new_idx = len(database)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    database[new_idx] = {
        'image': image,
        'id': id,
        'name': name,
        'encoding': encoding
    }
    with open(PKL_PATH, 'wb') as f:
        pkl.dump(database, f)
    return True
def get_info_from_id(id): 
    database = get_databse() 
    for idx, person in database.items(): 
        if person['id'] == id: 
            name = person['name']
            image = person['image']
            return name, image, idx       
    return None, None, None
def deleteOne(id):
    database = get_databse()
    id = str(id)
    for key, person in database.items():
        if person['id'] == id:
            del database[key]
            break
    with open(PKL_PATH,'wb') as f:
        pkl.dump(database,f)
    return True
def build_dataset():
    counter = 0
    for image in os.listdir(DATASET_DIR):
        image_path = os.path.join(DATASET_DIR,image)
        image_name = image.split('.')[0]
        parsed_name = image_name.split('_')
        person_id = parsed_name[0]
        person_name = ' '.join(parsed_name[1:])
        if not image_path.endswith('.jpg'):
            continue
        image = frg.load_image_file(image_path)
        information[counter]['image'] = image 
        information[counter]['id'] = person_id
        information[counter]['name'] = person_name
        information[counter]['encoding'] = frg.face_encodings(image)[0]
        counter += 1

    with open(os.path.join(DATASET_DIR,'database.pkl'),'wb') as f:
        pkl.dump(information,f)

if __name__ == "__main__": 
    deleteOne(4)

