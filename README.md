1. Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
plt.rcParams['figure.figsize'] = 10, 10
%matplotlib inline

2. Track the train file

train = pd.read_json("train.json")

Output the 5 first lines

train.head(5)

3.From json to numpy:

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])

Check the dimension:

X_band_1.shape

X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])


add one more axis:
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], 
                         axis=-1)

Check new  dimension:
X_train.shape

Lines with erroes:
[X_band_1[:, :, :, np.newaxis]].shape
(X_band_1+X_band_2)/2)[:, :, :, np.newaxis].shape

4. visualization with plotpy and output the one figure ( iceberg or ship)

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
def plotmy3d(c, name):

    data = [
        go.Surface(
            z=c
        )
    ]
    layout = go.Layout(
        title=name,
        autosize=False,
        width=700,
        height=700,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
plotmy3d(X_band_1[844,:,:], 'iceberg')