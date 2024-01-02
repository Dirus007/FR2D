import pickle
import os
from annoy import AnnoyIndex

f = 128


def load_data(encodings_file, mode, annoy_encodings_file):
    annoy_index = None
    if mode == 'normal':
        if os.path.exists(encodings_file):
            print("Loading face encodings...")
            data = pickle.loads(open(encodings_file, "rb").read())
        else:
            print("Starting with an empty dataset.")
            data = {"encodings": [], "service_number": [], "names": []}

    elif mode == 'annoy':
        if os.path.exists(encodings_file):
            print("Loading face encodings...")
            data = pickle.loads(open(encodings_file, "rb").read())
        else:
            print("Starting with an empty dataset.")
            data = {"encodings": [], "names": []}
            annoy_index = create_annoy_index(data, annoy_encodings_file)

    else:
        print("Incorrect mode")
        data = {"encodings": [], "service_number": [], "names":[]}

    return data, annoy_index

def create_annoy_index(data, annoy_encodings_file):
    global f
    new_annoy_index = AnnoyIndex(f, 'euclidean')
    for i, encoding in enumerate(data["encodings"]):
        new_annoy_index.add_item(i, encoding)
    new_annoy_index.build(10)
    index_path = os.path.join(os.getcwd(), annoy_encodings_file)
    new_annoy_index.save(index_path)
    return new_annoy_index