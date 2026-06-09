# Neural Network from scratch in Python

A feed-forward neural network implemented in **pure Python + NumPy**, with no
machine-learning framework — forward propagation, backpropagation, and gradient
descent written by hand. It trains on the classic
[Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg) dataset to predict a
car's fuel efficiency (miles per gallon) from features like cylinders, weight,
horsepower, and model year.

📝 **Write-up:** <https://jahnog.github.io/Artificial-Intelligence-Beginnings/>

## What's here

| File | Role |
|------|------|
| `models.py` | `FourLayerModel` — the network: ReLU hidden layers, linear output, and the forward/backward/update methods. |
| `data.py` | `DataSource` — downloads, cleans, one-hot encodes, normalizes, and splits the Auto MPG dataset. |
| `neural_network.py` | Training script — wires the model and data together and prints predictions vs. real values. |
| `neural_network.ipynb` | The same example as a runnable notebook. |

## Run it

```bash
pip install numpy pandas
python neural_network.py
```

Or open the notebook directly in your browser
([Google Colab](https://colab.research.google.com/github/jahnog/artificial-intelligence-python/blob/master/neural_network.ipynb)).

After ~15,000 training epochs the network's predictions land close to the real MPG
values — see the write-up for the full explanation of the math and code.

## Stack

Python · NumPy · pandas
