from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from PIL import Image
import io


class KerasModelWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(X)

# Load the pickled model
with open('traffic_classifier.pkl', 'rb') as f:
    model_wrapper = pickle.load(f)

# Class labels (shortened for brevity, use your full dict in production)
classes = classes = {
    0: ('Speed limit (20km/h)', 'Reduce your speed to 20 km/h.'),
    1: ('Speed limit (30km/h)', 'Reduce your speed to 30 km/h.'),
    2: ('Speed limit (50km/h)', 'Reduce your speed to 50 km/h.'),
    3: ('Speed limit (60km/h)', 'Reduce your speed to 60 km/h.'),
    4: ('Speed limit (70km/h)', 'Reduce your speed to 70 km/h.'),
    5: ('Speed limit (80km/h)', 'Reduce your speed to 80 km/h.'),
    6: ('End of speed limit (80km/h)', 'You may drive faster than 80 km/h where permitted.'),
    7: ('Speed limit (100km/h)', 'Increase your speed to 100 km/h if safe.'),
    8: ('Speed limit (120km/h)', 'Increase your speed to 120 km/h if safe.'),
    9: ('No passing', 'Do not overtake or pass other vehicles.'),
    10: ('No passing veh over 3.5 tons', 'Do not overtake vehicles over 3.5 tons.'),
    11: ('Right-of-way at intersection', 'Yield to vehicles coming from your right at intersections.'),
    12: ('Priority road', 'You have the priority at intersections; do not stop for vehicles on secondary roads.'),
    13: ('Yield', 'Slow down and give way to other vehicles or pedestrians.'),
    14: ('Stop', 'Come to a complete stop at the stop line, check for traffic, and proceed when safe.'),
    15: ('No vehicles', 'Do not enter this area with any vehicle.'),
    16: ('Veh > 3.5 tons prohibited', 'Do not allow vehicles weighing more than 3.5 tons to enter.'),
    17: ('No entry', 'Do not enter this road or area.'),
    18: ('General caution', 'Be cautious; watch for potential hazards ahead.'),
    19: ('Dangerous curve left', 'Slow down and prepare to turn left; drive carefully.'),
    20: ('Dangerous curve right', 'Slow down and prepare to turn right; drive carefully.'),
    21: ('Double curve', 'Be alert for a series of curves; adjust your speed accordingly.'),
    22: ('Bumpy road', 'Slow down; the road ahead is uneven or damaged.'),
    23: ('Slippery road', 'Reduce speed; be cautious of slippery conditions.'),
    24: ('Road narrows on the right', 'Be prepared for the road to narrow ahead; stay to the left if safe.'),
    25: ('Road work', 'Be alert for construction workers and equipment; reduce speed.'),
    26: ('Traffic signals', 'Respect traffic signals and stop when required.'),
    27: ('Pedestrians', 'Watch for pedestrians crossing; be prepared to stop.'),
    28: ('Children crossing', 'Be extra cautious; watch for children crossing the road.'),
    29: ('Bicycles crossing', 'Be alert for cyclists; give them space when they are on the road.'),
    30: ('Beware of ice/snow', 'Exercise caution; the road may be icy or snowy.'),
    31: ('Wild animals crossing', 'Slow down; be alert for animals crossing the road.'),
    32: ('End speed + passing limits', 'The previous speed and passing restrictions are no longer in effect.'),
    33: ('Turn right ahead', 'Prepare to turn right at the upcoming intersection.'),
    34: ('Turn left ahead', 'Prepare to turn left at the upcoming intersection.'),
    35: ('Ahead only', 'Proceed straight ahead; do not turn.'),
    36: ('Go straight or right', 'You can continue straight or turn right.'),
    37: ('Go straight or left', 'You can continue straight or turn left.'),
    38: ('Keep right', 'Stay in the right lane unless overtaking.'),
    39: ('Keep left', 'Stay in the left lane unless overtaking.'),
    40: ('Roundabout mandatory', 'You must enter the roundabout; yield to traffic inside the circle.'),
    41: ('End of no passing', 'You may overtake other vehicles where it is safe to do so.'),
    42: ('End no passing vehicle with a weight greater than 3.5 tons', 'You may overtake vehicles over 3.5 tons if safe to do so.')
}


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        image = Image.open(file.stream)
        image = image.resize((30, 30))
        image = np.array(image)
        if image.shape[-1] == 4:  # RGBA to RGB
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        pred = model_wrapper.predict(image)
        pred_class = int(np.argmax(pred, axis=1)[0])
        label = classes.get(pred_class, "Unknown")
        return jsonify({'prediction': label, 'class_id': pred_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)