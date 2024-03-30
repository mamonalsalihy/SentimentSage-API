import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os, openai

# Load the OpenAI API key from an environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("No OpenAI API key set.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)


# Load the dataset
data_path = 'data/Tweets.csv'
df = pd.read_csv(data_path)

X = df['text']
y = df['airline_sentiment']


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def target_map(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'neutral':
        return 0
    else:
        return -1


X.apply(lambda x: get_embedding)
y.apply(lambda x: target_map)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X)

# Initialize and train the RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save the model to disk
model_path = 'random_forest_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(rf, file)

# Save the test data to disk in CSV format
test_data_path = 'test_data.csv'
test_data = X_test.copy()
test_data['target'] = y_test  # Append the target column to your test features
test_data.to_csv(test_data_path, index=False)

print(f"Model saved to {model_path}")
print(f"Test data saved to {test_data_path}")