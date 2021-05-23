from models import FourLayerModel
from data import DataSource

X_train, Y_train, X_test, Y_test, desc = DataSource().get_training_data()

nn = FourLayerModel(X_train.shape[0], 19, 8)

TRAIN_EPOCHS = 20000
LEARNING_RATE = 0.001
PRINT_EACH = 100

for i in range(0, TRAIN_EPOCHS):
    test_predictions = nn.forward_propagation(X_test)
    test_cost = nn.calculate_cost(test_predictions, Y_test)

    predictions = nn.forward_propagation(X_train)
    cost = nn.calculate_cost(predictions, Y_train)

    nn.backward_propagation(X_train, Y_train)

    nn.update_weights(LEARNING_RATE)

    if i % PRINT_EACH == 0:
        print("Cost after iteration %i: %f %f" % (i, cost, test_cost))

test_predictions = nn.forward_propagation(X_test)

print("Predictions: ", test_predictions)
print("Real MPG:    ", Y_test)
print("Test Cost:   ", test_cost)
