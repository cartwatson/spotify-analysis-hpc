import pandas as pd
import time

class Instance():
    def __init__(self, danceability, acousticness, liveness):
        self.danceability: float = danceability
        self.acousticness: float = acousticness
        self.liveness: float = liveness

start = time.time()
file = open("tracks_features.csv", "r")
df = pd.read_csv(file, header=0, usecols=['danceability', 'acousticness', 'liveness'])

instances = []
for index, row in df.iterrows():
    instances.append(Instance(row['danceability'], row['acousticness'], row['liveness']))
    if (index % 100000 == 0):
        print(f"Read {index} rows")

print(f"Read {len(instances)} instances in {time.time() - start} seconds")
