import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split  # to split training and testing data
from keras.utils import to_categorical  # to convert the labels present in y_train and t_test into one-hot encoding
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout  # to create CNN

data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Retrieving the images and their labels
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(os.path.join(path, a))  # Corrected path joining
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print("Error loading image:", e)

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

# Splitting training and testing dataset
X_t1, X_t2, y_t1, y_t2 = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_t1.shape, X_t2.shape, y_t1.shape, y_t2.shape)

# Converting the labels into one hot encoding
y_t1 = to_categorical(y_t1, 43)
y_t2 = to_categorical(y_t2, 43)

# Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_t1.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
eps = 15
history = model.fit(X_t1, y_t1, batch_size=32, epochs=eps, validation_data=(X_t2, y_t2))
model.save("traffic_classifier.h5")  # Save the model

# Plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Testing accuracy on test dataset
from sklearn.metrics import accuracy_score
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)  # Get the class predictions
print("Test accuracy:", accuracy_score(labels, pred_classes))







# GUI code for image classification
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np

# Load the trained model to classify sign
model = load_model('traffic_classifier.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Dictionary to label all traffic signs class.
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



# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Classification')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = np.expand_dims(np.array(image), axis=0)  # Expand dimensions for model input
    pred = model.predict(image)
    pred_classes = np.argmax(pred, axis=1)  # Get the class prediction
    sign = classes[pred_classes[0]]
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print("Error uploading image:", e)

upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text="Check Traffic Sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()







""" from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Assuming `labels` are the true labels from your test data
# and `pred_classes` are the predicted labels from your model

# Generate a classification report
report = classification_report(labels, pred_classes, output_dict=True)

# Convert the report to a DataFrame for a tabular display
report_df = pd.DataFrame(report).transpose()

# Display overall accuracy
accuracy = accuracy_score(labels, pred_classes)
print("Overall Test Accuracy:", accuracy)

# Display the report in a table format for readability
print(report_df)

# Save the report as a CSV or Excel file if desired
report_df.to_csv("traffic_sign_classification_report.csv") """
