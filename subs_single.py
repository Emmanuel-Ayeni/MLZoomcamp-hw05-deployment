
# Ensure Pipenv steps are followed: 
    # 1. Ensure pipenv is installed. You can install it via pip: Bash: $pip install pipenv.
    # 2, Create a New Project Directory:$ mkdir my_data_science_project, $ cd my_data_science_project
    # 3. Initialize a Pipfile and virtual environment with pipenv. To specify Python version (recommended), you can use: 
         #Bash $pipenv --python 3.9  # Replace with your preferred Python version
         # $ pipenv --python 3.9  # Replace with your preferred Python version





import pickle 

# Load the vectorizer and model
with open("dv.bin", "rb") as dv_file:
    dv = pickle.load(dv_file)

with open("model1.bin", "rb") as model_file:
    model = pickle.load(model_file)

# Define client data
client_data = {"job": "management", "duration": 400, "poutcome": "success"}

# Transform the client data
X_client = dv.transform([client_data])

# Score the client
score = model.predict(X_client)[0]
print("Client Score:", score)
