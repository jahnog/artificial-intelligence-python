from models import FourLayerModel
from data import DataSource

X_train, Y_train, X_test, Y_test, desc = DataSource().get_training_data()

TRAIN_EPOCHS = 20001
LEARNING_RATE = 0.001
PRINT_EACH = 10000

for h1 in range(8, 9):
    for h2 in range(5, 6):

        for l in range(10):
            min_epoch = 0
            min_cost = 0
            min_test_cost = 0

            print("-----------------------------------------------------------------------")
            print("h1: ", h1, " - h2: ", h2)

            nn = FourLayerModel(X_train.shape[0], h1, h2)

            for i in range(0, TRAIN_EPOCHS):
                test_predictions = nn.forward_propagation(X_test)
                test_cost = nn.calculate_cost(test_predictions, Y_test)

                predictions = nn.forward_propagation(X_train)
                cost = nn.calculate_cost(predictions, Y_train)

                nn.backward_propagation(X_train, Y_train)

                nn.update_weights(LEARNING_RATE)

                if (i) % PRINT_EACH == 0:
                    print("Cost after iteration %i: %f %f" % (i, cost, test_cost))

                if i == 0:
                    min_cost = cost
                    min_test_cost = test_cost
                else:
                    if test_cost < min_test_cost:
                        min_epoch = i
                        min_cost = cost
                        min_test_cost = test_cost

            print("Min epoch: ", min_epoch, " - cost: ",
                min_cost, "- test cost: ", min_test_cost)
