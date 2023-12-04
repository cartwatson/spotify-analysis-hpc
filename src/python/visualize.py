import pandas as pd
import numpy as np
from vispy import app, scene, color, visuals
from pathlib import Path

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

camera = scene.TurntableCamera(fov=60)
camera.center = (0.5, 0.5, 0.5)

view.camera = camera

scatter = scene.visuals.Markers()

# Set scatter data
pos = np.column_stack((danceability, acousticness, liveness))
colors = color.get_colormap('autumn').map(cluster / cluster.max())
scatter.set_data(pos=pos, edge_color=None, face_color=colors, size=7)

# Add scatter plot to the view
view.add(scatter)

# Add a 3D axis
xax = scene.Axis(
    pos=[[0, 0], [1, 0]],
    tick_direction=(0, -1),
    axis_color='black',
    tick_color='black',
    text_color='black',
    font_size=12,
    axis_label='danceability',
    parent=view.scene,
)
yax = scene.Axis(
    pos=[[0, 0], [0, 1]],
    tick_direction=(-1, 0),
    axis_color='black',
    tick_color='black',
    text_color='black',
    font_size=12,
    axis_label='acousticness',
    parent=view.scene,
)
zax = scene.Axis(
    pos=[[0, 0], [-1, 0]],
    tick_direction=(0, -1),
    axis_color='black',
    tick_color='black',
    text_color='black',
    font_size=12,
    axis_label='liveness',
    parent=view.scene,
)
zax.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

# Run the application
app.run()
