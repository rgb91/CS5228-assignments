"""
Created by Sanjay at 10/28/2018

Feature: Enter feature name here
Enter feature description here
"""
import numpy as np
import os
import json
import operator

from sklearn.tree import DecisionTreeRegressor


class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=1):
        """
        Initialization
        :param max_depth: type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split: type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        """
        pass

    def predict(self, X):
        """
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        """
        pass

    def get_model_string(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)

def func():
    l1 = list ([1, 2, 3])
    l2 = list ([6, 7, 8])
    return l1+ l2

# For test
if __name__=='__main__':

    x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(0) +".csv", delimiter=",")
    y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(0) +".csv", delimiter=",")
    y_train = np.reshape(y_train, (y_train.shape[0], 1))

    # Decision Tree from sk-learn library
    i, j = 0, 0
    regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=j+2, random_state=0)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_train)

    from sklearn.externals.six import StringIO
    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import pydotplus

    dot_data = StringIO ()

    export_graphviz (regressor, out_file=dot_data,
                     filled=True, rounded=True,
                     special_characters=True)

    graph = pydotplus.graph_from_dot_data (dot_data.getvalue ())
    Image (graph.create_png ())

    # for j in range(2):
    #     tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
    #     tree.fit(x_train, y_train)
    #
    #     model_string = tree.get_model_string()
    #
    #     with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
    #         test_model_string = json.load(fp)
    #
    #     print(operator.eq(model_string, test_model_string))
    #
    #     y_pred = tree.predict(x_train)
    #
    #     y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")
    #     print(np.square(y_pred - y_test_pred).mean() <= 10**-10)

