import pickle
from keras.models import load_model  # Add this import

# Load your trained model
model = load_model('traffic_classifier.h5')

# Define a wrapper class
class KerasModelWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(X)

# Wrap the model
wrapped_model = KerasModelWrapper(model)

# Save the wrapped model as a .pkl file
with open('traffic_classifier.pkl', 'wb') as f:
    pickle.dump(wrapped_model, f, protocol=pickle.HIGHEST_PROTOCOL)  # Use highest protocol

print("Model saved as traffic_classifier.pkl")