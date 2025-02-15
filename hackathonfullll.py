import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
df = pd.read_csv(r"C:\Users\SHREYA\Downloads\sbdb_query_results.csv")

# Ensure all required columns exist
required_columns = ['full_name', 'diameter', 'albedo', 'GM', 'i', 'class']
for col in required_columns:
    if col not in df.columns:
        print(f"Error: Missing column '{col}' in dataset!")
        exit()

# Fill missing values correctly
df.fillna({"albedo": df['albedo'].median(), "GM": df['GM'].median(), "i": df['i'].median()}, inplace=True)

# Constants for impact energy calculations
ASTEROID_DENSITY = 3000  # kg/m³ (rocky asteroid)
IMPACT_VELOCITY = 20  # km/s (typical impact speed)

# Function to clean asteroid names
def clean_name(name):
    parts = name.strip().split()
    if parts[0].isdigit():
        return " ".join(parts[1:]).strip()
    return name.strip()

# Apply name cleaning and standardize names for matching
df['clean_name'] = df['full_name'].apply(clean_name).str.lower().str.strip()

# Features for ML model
features = ['diameter', 'albedo', 'GM', 'i']

# Standardize feature values
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Train Random Forest model for risk classification
X = df[features]
y = df['class']
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Function to calculate impact energy in megatons
def calculate_impact_energy(diameter_km, density=ASTEROID_DENSITY, velocity=IMPACT_VELOCITY):
    radius_m = (diameter_km * 1000) / 2
    mass = (4/3) * np.pi * radius_m**3 * density  # Mass in kg
    energy_joules = 0.5 * mass * (velocity * 1000) ** 2
    return energy_joules / (4.184e15)  # Convert Joules to megatons

# Function to predict disasters based on impact energy
def predict_disasters(energy_megatons):
    if energy_megatons < 1:
        return ["Minimal surface damage"]
    elif energy_megatons < 10:
        return ["Local earthquake (Magnitude 4-5)", "Small airburst explosion"]
    elif energy_megatons < 100:
        return ["Severe earthquake (Magnitude 6-7)", "Local tsunami", "Firestorms"]
    elif energy_megatons < 1000:
        return ["Major tsunami", "Atmospheric shockwave", "Massive earthquakes (Magnitude 8+)"]
    else:
        return ["Global firestorms", "Climate change", "Worldwide tsunami"]

# Function to classify asteroid
def classify_asteroid(row):
    gm = row.get('GM', np.nan)
    albedo = row.get('albedo', np.nan)
    if pd.isna(gm) or pd.isna(albedo):
        return "Unknown"
    if gm > 5 and albedo > 0.15:
        return "Metal-Rich"
    elif gm > 1 and albedo > 0.08:
        return "Stony"
    else:
        return "Carbonaceous"

# Function to classify as Silicate or Rocky
def classify_silicate_rocky(row):
    inclination = row.get('i', np.nan)
    return "Silicate" if inclination > 10 else "Rocky"

# Function to estimate mining potential
def mining_potential(row, classification):
    gm = row.get('GM', np.nan)
    albedo = row.get('albedo', np.nan)
    if classification == "Metal-Rich":
        return "Very High (Platinum, Nickel, Rare Earths)" if gm > 10 else "High (Iron, Nickel, Platinum)"
    elif classification == "Stony":
        return "Moderate (Silicates, Magnesium, Olivine)" if albedo > 0.12 else "Low (Silica, Basalt)"
    elif classification == "Carbonaceous":
        return "Very Low (Carbon, Water, Organics)"
    return "Unknown"

# Function to calculate risk percentage
def calculate_risk(row):
    diameter_km = row.get('diameter', np.nan)
    gm = row.get('GM', np.nan)
    if pd.isna(diameter_km) or pd.isna(gm):
        return 0.0
    risk = (diameter_km * 0.05 + gm * 0.03) * 10
    return min(risk, 100)  # Cap risk at 100%

# Function to predict asteroid details
def predict_asteroid_details(asteroid_name):
    # Try exact match for asteroid name
    asteroid_data = df.loc[df['clean_name'].str.contains(asteroid_name.lower(), case=False, regex=False)]
    
    if asteroid_data.empty:
        print(f"No data found for asteroid: {asteroid_name}")
        return

    asteroid_data = asteroid_data.iloc[0]

    # Calculate impact energy
    diameter_km = asteroid_data['diameter']
    impact_energy = calculate_impact_energy(diameter_km)

    # Predict disasters based on impact energy
    predicted_disasters = predict_disasters(impact_energy)

    # Classification and mining potential
    classification = classify_asteroid(asteroid_data)
    silicate_class = classify_silicate_rocky(asteroid_data)
    mining_value = mining_potential(asteroid_data, classification)

    # Risk percentage
    risk_percentage = calculate_risk(asteroid_data)

    # Feature extraction and prediction
    asteroid_features = asteroid_data[features].values.reshape(1, -1)
    asteroid_scaled = scaler.transform(asteroid_features)
    risk_class = rf_model.predict(asteroid_scaled)

    # Display results
    print("\n--- Asteroid Details ---")
    print(f"Asteroid Name: {asteroid_data['full_name']}")
    print(f"Impact Energy: {impact_energy:.2f} Megatons")
    print(f"Predicted Disasters: {', '.join(predicted_disasters)}")
    print(f"Predicted Class: {risk_class[0]}")
    print(f"Classification: {classification}")
    print(f"Silicate/Rocky Type: {silicate_class}")
    print(f"Mining Potential: {mining_value}")
    print(f"Risk Percentage: {risk_percentage:.2f}%")

# Get user input and predict asteroid details
asteroid_name = input("Enter asteroid name: ").strip()
predict_asteroid_details(asteroid_name)
