from models import FourLayerModel
from data import DataSource

X_train, Y_train, X_test, Y_test, desc = DataSource().get_training_data()

nn = FourLayerModel(X_train.shape[0], 12, 5)

TRAIN_EPOCHS = 5000
LEARNING_RATE = 0.001
PRINT_EACH = 100

for i in range(0, TRAIN_EPOCHS):
    prediction = nn.forward_propagation(X_train)

    cost = nn.calculate_cost(prediction, Y_train)

    nn.backward_propagation(X_train, Y_train)

    nn.update_weights(LEARNING_RATE)

    # Print the cost every 1000 iterations
    if i % PRINT_EACH == 0:
        print("Cost after iteration %i: %f" % (i, cost))
