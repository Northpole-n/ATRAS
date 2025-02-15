import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

def preprocess_data(file_path):
    """Load and preprocess the asteroid dataset."""
    df = pd.read_csv(file_path)
    
    # Rename MOID column to distance for consistency
    df.rename(columns={'moid': 'distance'}, inplace=True)
    
    # Select relevant features
    features = ['neo', 'diameter', 'H', 'distance', 'albedo', 'name']
    df = df[features].dropna()
    
    # Encode categorical variables
    le = LabelEncoder()
    df['neo'] = le.fit_transform(df['neo'])  # Y/N to 1/0
    
    # Normalize numerical features using MinMaxScaler
    scaler = MinMaxScaler()
    df[['diameter', 'H', 'distance', 'albedo']] = scaler.fit_transform(df[['diameter', 'H', 'distance', 'albedo']])
    
    # Define target variable (custom risk calculation instead of PHA)
    df['risk_score'] = df['diameter'] * 0.4 + df['H'] * 0.6  # Example risk formula
    df['risk_score'] = df['risk_score'].clip(0, 1)  # Ensure values are between 0 and 1
    
    # Split data into train and test sets
    X = df.drop(columns=['risk_score', 'name'])  # Features
    y = df['risk_score']  # Target variable (continuous risk score)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return df, X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a Random Forest model on the asteroid data."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def analyze_risk(df, model, asteroid_name):
    """Analyze the risk percentage of a given asteroid by name."""
    asteroid = df[df['name'].str.lower() == asteroid_name.lower()]
    if asteroid.empty:
        return "Asteroid not found."
    
    X_input = asteroid[['neo', 'diameter', 'H', 'distance', 'albedo']]
    
    # Predict risk percentage and clip to 0-100%
    risk_score = model.predict(X_input)[0] * 100  # Scale to percentage
    risk_score = max(0, min(risk_score, 100))  # Ensure it stays within valid range
    
    return (f"Risk percentage for {asteroid_name}: {risk_score:.2f}%\n"
            f"Distance from Earth: {asteroid['distance'].values[0]:.2f} AU\n"
            f"Albedo: {asteroid['albedo'].values[0]:.2f}")

# Enter the file name or full path here
df, X_train, X_test, y_train, y_test = preprocess_data("darkast.csv")  # Ensure correct path

model = train_model(X_train, y_train)  # Train the model

# Take asteroid name as input from the user
asteroid_name = input("Enter the asteroid name: ")
print(analyze_risk(df, model, asteroid_name))
