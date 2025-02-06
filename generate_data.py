import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
data = {
    "Age": np.random.randint(18, 65, 100),
    "Salary": np.random.randint(30000, 120000, 100),
    "Experience": np.random.randint(0, 40, 100),
    "Satisfaction": np.random.uniform(1, 5, 100)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV file
csv_filename = "./data/data.csv"
df.to_csv(csv_filename, index=False)

csv_filename
