import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pulp

# Fetch data from FPL API
base_url = "https://fantasy.premierleague.com/api/"
bootstrap_url = base_url + "bootstrap-static/"

response = requests.get(bootstrap_url)
json_data = response.json()

# Extract players' data
players = json_data['elements']

# Convert the data to a DataFrame
df = pd.DataFrame(players)

# Preprocess
df.fillna(0, inplace=True)


# Feature engineering
df['points_per_million'] = df['total_points'] / df['now_cost']

# Model training
features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'points_per_million']
X = df[features]
y = df['total_points']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# Optimization
def select_team(data, budget=1000):
    prob = pulp.LpProblem('FPL_Team_Selection', pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("Player", data.index, cat='Binary')
    
    prob += pulp.lpSum(data['predicted_points'][i] * player_vars[i] for i in data.index), "Total Predicted Points"
    
    prob += pulp.lpSum(data['now_cost'][i] * player_vars[i] for i in data.index) <= budget, "Total Cost"
    # Further constraints can be added as required
    
    prob.solve()
    
    selected_indices = [v.name for v in prob.variables() if v.varValue == 1]
    return data.loc[[int(i.split("_")[1]) for i in selected_indices]]

df['predicted_points'] = model.predict(X)
best_team = select_team(df)
print(best_team)

