import pandas as pd
import numpy as np
from vispy import app, scene, color
from pathlib import Path
import sys

filepath = Path(__file__).parent / '../data/output.csv' 
df = pd.read_csv(filepath)

danceability = df['danceability'].values
acousticness = df['acousticness'].values
liveness = df['liveness'].values
cluster = df['cluster'].values

# Create a canvas
canvas = scene.SceneCanvas(keys='interactive', bgcolor='#cccccc', size=(1000, 600), show=True)

# Create a 3D scatter plot view
view = canvas.central_widget.add_view()
scatter = scene.visuals.Markers()

# Set scatter data
pos = np.column_stack((danceability, acousticness, liveness))
colors = color.get_colormap('autumn').map(cluster / cluster.max())
scatter.set_data(pos=pos, edge_color=None, face_color=colors, size=5)

# Add scatter plot to the view
view.add(scatter)

# Set camera parameters
view.camera = 'turntable'
view.camera.fov = 90

# Run the application
app.run()
