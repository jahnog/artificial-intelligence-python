from models import FourLayerModel
from data import DataSource

X_train, Y_train, X_test, Y_test, desc = DataSource().get_training_data()

TRAIN_EPOCHS = 15001
LEARNING_RATE = 0.001
PRINT_EACH = 1000

nn = FourLayerModel(X_train.shape[0], 5, 4)

for i in range(0, TRAIN_EPOCHS):
    predictions = nn.forward_propagation(X_train)
    cost = nn.calculate_cost(predictions, Y_train)

    nn.backward_propagation(X_train, Y_train)

    nn.update_weights(LEARNING_RATE)

    if (i) % PRINT_EACH == 0:
        print("Cost after iteration %i: %f " % (i, cost))

test_predictions = nn.forward_propagation(X_test)

print("Predictions: ", test_predictions.astype(int))
print("Real Values: ", Y_test.astype(int))
