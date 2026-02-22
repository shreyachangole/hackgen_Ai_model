import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

num_bins = 50
days = 30
records = []

location_types = {
    "Residential": 1.0,
    "Commercial": 2.0,
    "Industrial": 1.5
}

start_date = datetime.now()

for bin_id in range(1, num_bins + 1):

    location = random.choice(list(location_types.keys()))
    fill_rate = location_types[location]
    fill_percent = random.randint(0, 20)

    for day in range(days):
        for hour in range(0, 24, 6):

            timestamp = start_date + timedelta(days=day, hours=hour)

            weekend_multiplier = 1.3 if timestamp.weekday() >= 5 else 1.0

            increase = np.random.uniform(5, 15) * fill_rate * weekend_multiplier
            fill_percent = min(fill_percent + increase, 100)

            if fill_percent < 100:
                hours_to_full = (100 - fill_percent) / (increase / 6)
            else:
                hours_to_full = 0

            records.append([
                bin_id,
                timestamp,
                location,
                round(fill_percent, 2),
                1 if timestamp.weekday() >= 5 else 0,
                round(hours_to_full, 2)
            ])

            if fill_percent >= 100:
                fill_percent = random.randint(0, 10)

columns = [
    "bin_id",
    "timestamp",
    "location_type",
    "fill_percent",
    "is_weekend",
    "hours_to_full"
]

df = pd.DataFrame(records, columns=columns)
df.to_csv("simulated_50_bins.csv", index=False)

print("✅ 50 Bins - 1 Month Data Generated")