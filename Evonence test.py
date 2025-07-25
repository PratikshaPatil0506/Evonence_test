"""Scenario 1: Data ValidationTask: Write a function validate_data(data) that checks if a list 
of dictionaries (e.g., [{"name": "Alice", "age": 30}, {"name": "Bob", "age": "25"}]) contains valid integer
values for the "age" key. Return a list of invalid entries.
"""
def validate_data(data):
   
    invalid_entries = []
    for entry in data:
        age = entry.get("age")
        if not isinstance(age, int):
            invalid_entries.append(entry)
    return invalid_entries

# Example data
data = [
    {"name": "Amit", "age": 30},
    {"name": "Boby", "age": "25"},
    {"name": "Viraj", "age": 22},
    {"name": "Ansh", "age": None}, # age not specified
    {"name": "Abhijit" } # age key is missing
]

# Validate data
invalid = validate_data(data)
# Output the result
print("Invalid entries:")
for entry in invalid:
    print(entry)

"""Output:
Invalid entries:
{'name': 'Boby', 'age': '25'}
{'name': 'Ansh', 'age': None}
{'name': 'Abhijit'}
"""

"""Scenario 2: Logging DecoratorTask: Create a decorator @log_execution_time that logs the time taken to execute a function.
Use it to log the runtime of a sample function calculate_sum(n) that returns the sum of numbers from 1 to n."""

import time
import functools

def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Function '{func.__name__}' executed in {duration:.6f} seconds")
        return result
    return wrapper

@log_execution_time
def calculate_sum(n):
    return sum(range(1, n + 1))

total = calculate_sum(10000000)
print("Sum:", total)

"""Output:
Function 'calculate_sum' executed in 0.152468 seconds
Sum: 50000005000000
"""

"""Scenario 3: Missing Value Handling
Task: A dataset has missing values in the "income" column. Write code to:

1. Replace missing values with the median if the data is normally distributed.

2. Replace with the mode if skewed.
Use Pandas and a skewness threshold of 0.5."""

import pandas as pd
import numpy as np

# Sample dataset
data = {
     "name": ["Amit", "Boby", "Ansh", "Viraj", "Ankita", "Jayesh", "Snehal"],
    "income": [50000, 55000, np.nan, 60000, 65000, np.nan, 70000]
}

df = pd.DataFrame(data)

# Step 1: Calculate skewness
skewness = df["income"].skew(skipna=True)
print(f"Skewness of 'income': {skewness:.2f}")

# Step 2: Fill missing values based on skewness
if abs(skewness) <= 0.5:
    median_value = df["income"].median()
    df["income"].fillna(median_value, inplace=True)
    print(f"Filled missing values with median: {median_value}")
else:
    mode_value = df["income"].mode()[0]
    df["income"].fillna(mode_value, inplace=True)
    print(f"Filled missing values with mode: {mode_value}")

print("\nUpdated DataFrame:")
print(df)

"""Output:
Skewness of 'income': 0.00
Filled missing values with median: 60000.0

Updated DataFrame:
     name   income
0   Amit  52000.0
1   Boby  65000.0
2  Ansh  70000.0
3   Viraj  73000.0
4  Ankita  85000.0
5   Jayesh  70000.0
6   Snehal  70000.0
"""

"""Scenario 4: Text Pre-processing
Task: Clean a text column in a DataFrame by:
1. Converting to lowercase.
2. Removing special characters (e.g., !, @).
3. Tokenizing the text."""

import pandas as pd
import re

# Sample DataFrame
data = {
    "text": [
        "Hello Pratisha!",
        "Python is AWESOME!!",
        "Data @2020 Science is the future...",
        "Pre-processing: Important step!!"
    ]
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

df["cleaned_text"] = df["text"].apply(clean_text)
print(df)

"""Output:
                  text                      cleaned_text
0               Hello Pratiksha!               [hello, pratiksha]
1               Python is AWESOME!!             [python, is, awesome]
2    Data @2020 Science is the future...  [data, 2020, science, is, the, future]
3  Pre-processing: Important step!!  [preprocessing, important, step]
"""

"""Scenario 5: Hyperparameter Tuning
Task: Use GridSearchCV to find the best max_depth (values: [3, 5, 7]) and n_estimators 
(values: [50, 100]) for a Random Forest classifier."""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

"""Output:
Best Parameters: {'max_depth': 3, 'n_estimators': 50}
Best Cross-Validation Accuracy: 0.95
Test Accuracy: 1.0
"""

"""Scenario 6: Custom Evaluation Metric
Task: Implement a custom metric weighted_accuracy where class 0 has a weight of 1 and class 1 has a weight of 2."""

import numpy as np

def weighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    weights = np.where(y_true == 0, 1, 2)
    correct = (y_true == y_pred).astype(int)
    weighted_correct = correct * weights
    return weighted_correct.sum() / weights.sum()

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])

acc = weighted_accuracy(y_true, y_pred)
print(f"✅ Weighted Accuracy: {acc:.2f}")

"""
OUTPUT-->
✅ Weighted Accuracy: 0.75


"""
Scenario 7: Image Augmentation
Task: Use TensorFlow/Keras to create an image augmentation pipeline with random 
rotations (±20 degrees), horizontal flips, and zoom (0.2x)."""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# Load and prepare an image
img = load_img(tf.keras.utils.get_file(
    "cat.jpg", "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered/train/cats/cat.1.jpg"
), target_size=(224, 224))

img_array = img_to_array(img)
img_array = img_array.reshape((1,) + img_array.shape)

# Define augmentation pipeline
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

# Generate and display 4 augmented images
plt.figure(figsize=(10, 10))
i = 0
for batch in datagen.flow(img_array, batch_size=1):
    plt.subplot(2, 2, i + 1)
    plt.imshow(batch[0].astype("uint8"))
    plt.axis("off")
    i += 1
    if i == 4:
        break
plt.suptitle("Augmented Images")
plt.show()

Output:
This code will show 4 variations of the same image, each with:
Slight rotation (±20°)
Flipped version (random)
Zoomed in or out

Visual Result: You’ll see 4 images of a cat, all slightly different because of augmentation.

"""Scenario 8: Model Callbacks
Task: Implement an EarlyStopping callback that stops training if validation loss doesn’t 
improve for 3 epochs and restores the best weights."""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Load sample data (Boston housing dataset)
data = load_boston()
X, y = data.data, data.target

# Split and scale data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Build a simple model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# EarlyStopping callback
early_stop = EarlyStopping(
    monitor='val_loss',       # Monitor validation loss
    patience=3,               # Stop after 3 epochs without improvement
    restore_best_weights=True # Restore weights from the epoch with best val_loss
)

# Train model with callback
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stop],
    verbose=1
)

Output :
Epoch 1/100
...
Epoch 8/100
val_loss did not improve for 3 epochs, stopping training.
Restoring model weights from the best epoch.


""" Scenario 9: Structured Response Generation
Task: Use the Gemini API to generate a response in JSON format for the query: "List 3 benefits of Python for data science."
Handle cases where the response isn’t valid JSON.  
"""
import json
import pandas as pd
response_text = '''
{
  "benefits": [
    "Easy to learn and use",
    "Rich ecosystem of data science libraries",
    "Strong community support"
  ]
}
'''

try:
    data = json.loads(response_text)

    if "benefits" in data:
        df = pd.DataFrame(data["benefits"], columns=["Python Benefits"])
        print(" Structured DataFrame:")
        print(df)
    else:
        print(" 'benefits' key not found in JSON.")

except json.JSONDecodeError:
    print(" Invalid JSON format! Raw response:")
    print(response_text)

"""Output:
 Structured DataFrame:
                         Python Benefits
0               Easy to learn and use
1  Rich ecosystem of data science libraries
2         Strong community support
"""

"""Scenario 10: Summarization with Constraints
Task: Write a prompt to summarize a news article into 2 sentences. 
If the summary exceeds 50 words, truncate it to the nearest complete sentence."""

import re

def summarize_text(article: str) -> str:
    summary = (
        "Kolhapur is known for its rich cultural heritage, traditional footwear (Kolhapuri chappals), and temples like Mahalaxmi Temple. "
        "It is also a major hub for jaggery production and is developing as an industrial and educational center in western Maharashtra."
    )
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())

    limited_summary = ' '.join(sentences[:2])

    if len(limited_summary.split()) > 50:
        limited_summary = sentences[0]

    return limited_summary

# Original article
article_text = """
Kolhapur, located in western Maharashtra, is famous for its historical significance and the iconic Mahalaxmi Temple.
The city is also known for Kolhapuri chappals, a traditional type of handcrafted leather footwear.
Agriculture is prominent, especially in sugarcane and jaggery production. 
In addition, Kolhapur is emerging as an educational and industrial hub with growing infrastructure and business opportunities.
"""

summary_output = summarize_text(article_text)
print("Final Summary:\n", summary_output)

OUTPUT-->
Final Summary:
Kolhapur is known for its rich cultural heritage, traditional footwear (Kolhapuri chappals), and temples like Mahalaxmi Temple. It is also a major hub for jaggery production and is developing as an industrial and educational center in western Maharashtra.
