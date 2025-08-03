
import pandas as pd

# Define mock data
data = pd.DataFrame({
    'num_layers': [2, 4, 6, 8, 10],
    'training_hours': [1, 2, 3, 4, 5],
    'flops_per_hour': [10, 20, 40, 60, 90],
    'energy_consumption': [0.5, 1.3, 2.8, 4.5, 6.9]  # Renamed to match your model's expected column
})

# Save to CSV
csv_path = r"C:\Users\Jasmine\PycharmProjects\New_project\data\synthetic\energy_data.csv"
data.to_csv(csv_path, index=False)

print(f"✅ Synthetic data saved to: {csv_path}")