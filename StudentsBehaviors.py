import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import graphviz
import warnings

# Disable all warnings
warnings.filterwarnings('ignore')

# Import the dataset
data = pd.read_csv('overdrawn.csv')

# Recategorize 'DaysDrink' into categorical data
conditions = [
    (data['DaysDrink'] < 7),
    (data['DaysDrink'] >= 14),
    (data['DaysDrink'] >= 7) & (data['DaysDrink'] < 14)
]
labels = [0, 2, 1]
data['DaysDrink'] = np.select(conditions, labels)

# Define features and target
features = data[['Age', 'Sex', 'DaysDrink']]
target = data['Overdrawn']

# Divide data into train and test subsets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train a Decision Tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(features_train, target_train)

# Predict the test dataset
predictions = classifier.predict(features_test)

# Compute the confusion matrix
conf_matrix = confusion_matrix(target_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Evaluate accuracy
accuracy = accuracy_score(target_test, predictions)
print("Accuracy:", accuracy)

# Visualize the decision tree
export_graphviz(classifier, out_file='tree.dot', 
                feature_names=features.columns,
                class_names=['No', 'Yes'],  
                filled=True, rounded=True,
                special_characters=True)

# Convert DOT file to PNG image with Graphviz
with open("tree.dot") as file:
    dot_graph = file.read()
graphviz.Source(dot_graph).render("overdrawn_dt", format="png", cleanup=True)

# Function for scenario predictions
def predict_scenarios(model, scenarios):
    results = []
    for scenario in scenarios:
        age, sex, days_drink = scenario
        result = model.predict([[age, sex, days_drink]])
        results.append(result[0])
    return results

# Test scenarios
test_scenarios = [
    (20, 0, 10),  # 20-year-old male, drinks 10 days
    (25, 1, 5),   # 25-year-old female, drinks 5 days
    (19, 0, 20),  # 19-year-old male, drinks 20 days
    (22, 1, 15),  # 22-year-old female, drinks 15 days
    (21, 0, 20)   # 21-year-old male, drinks 20 days
]

# Generate predictions for the scenarios
scenario_predictions = predict_scenarios(classifier, test_scenarios)

# Output the predictions
for index, prediction in enumerate(scenario_predictions, start=1):
    print(f"Prediction {index}: Will the student overdraw a checking account? {'Yes' if prediction == 1 else 'No'}")
