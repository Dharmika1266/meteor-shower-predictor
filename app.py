import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('static/meteor_showers_100.csv')

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Extract 'Year' and 'Month' from the 'Date'
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Optionally categorize meteor rates (if applicable)
def categorize_rate(rate):
    if rate > 80:
        return 'High'
    elif rate > 20:
        return 'Medium'
    else:
        return 'Low'

data['Rate_Category'] = data['Meteor_Rate'].apply(categorize_rate)

# Features (Month, Year) and target (Shower_Type)
X = data[['Month', 'Year']]
y = data['Shower_Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction function
def predict_meteor_shower(input_year, input_month):
    user_input = pd.DataFrame({'Month': [input_month], 'Year': [input_year]})
    prediction = model.predict(user_input)
    return prediction[0]

# Function to create graphs
def create_graphs():
    # Generate meteor shower data by month
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Month', data=data)
    plt.title('Meteor Showers by Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.savefig('static/meteor_shower_by_month.png')  # Save the figure
    plt.close()  # Close the figure to prevent display

    # Generate meteor shower data by year
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Year', data=data)
    plt.title('Meteor Showers by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.savefig('static/meteor_shower_by_year.png')  # Save the figure
    plt.close()  # Close the figure to prevent display

# Create Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        input_year = int(request.form['year'])
        input_month = int(request.form['month'])
        prediction = predict_meteor_shower(input_year, input_month)
        create_graphs()  # Call to create graphs
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use a different port

