import pickle

# Specify the path to your .pkl file
file_path = "./calibration.pkl"

# Open the file in binary mode
with open(file_path, "rb") as file:
    # Load the object from the file
    loaded_object = pickle.load(file)

# Now you can use the loaded_object
print("Loaded object:", loaded_object)
