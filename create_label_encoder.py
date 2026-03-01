import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# Create directory if it doesn't exist
os.makedirs("./Normal disease model", exist_ok=True)

# Based on your dataset, create label encoder with actual diseases
diseases = ["Influenza", "Common Cold", "Eczema", "Asthma", "Bronchitis", 
            "Pneumonia", "Allergic Rhinitis", "Sinusitis"]

label_encoder = LabelEncoder()
label_encoder.fit(diseases)

# Save the label encoder
joblib.dump(label_encoder, "./Normal disease model/label_encoder.pkl")

print("Label encoder created successfully!")
print("\nDisease Mappings:")
for i, disease in enumerate(diseases):
    print(f"{i}: {disease}")
