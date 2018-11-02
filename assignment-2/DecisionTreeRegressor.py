import numpy as np
import os
import json
import operator


def get_test_split(index, value, dataset_np):
    split = dict ()
    split['splitting_variable'] = index
    split['splitting_threshold'] = value
    temp_left = (dataset_np[:, index] <= value)
    temp_right = (dataset_np[:, index] > value)
    left_parition = dataset_np[np.nonzero (temp_left)]
    right_parition = dataset_np[np.nonzero (dataset_np[:, index] > value)]
    left_mean = np.nanmean (left_parition[:, -1])
    right_mean = np.nanmean (right_parition[:, -1])
    split['left'] = left_mean
    split['right'] = right_mean

    y_pred = (left_mean * temp_left) + (right_mean * temp_right)

    partitions = [left_parition, right_parition]

    return split, partitions, y_pred


class MyDecisionTreeRegressor ():
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

    def get_best_split(self, dataset):
        min_loss, best_partition, best_split = 999999, [], {}
        X_values = dataset[:, :-1]
        y_values = dataset[:, -1]
        for index in range (X_values.shape[1]):
            for value in X_values[:, index]:
                temp_split, temp_partitions, temp_y_pred = get_test_split (index, value, dataset)
                loss = np.sum (np.square (temp_y_pred - y_values))  # Calculation of loss
                if loss < min_loss:
                    min_loss = loss
                    best_partition = temp_partitions
                    best_split = temp_split
        return best_split, best_partition

    def build_tree(self, dataset, current_depth):
        # Base case
        if current_depth == self.max_depth or len (dataset) < self.min_samples_split:
            y_values = dataset[:, -1]
            return np.mean (y_values)
        node, partitions = self.get_best_split (dataset)

        node['left'] = self.build_tree (partitions[0], current_depth=current_depth + 1)  # partitions[0] == left
        node['right'] = self.build_tree (partitions[1], current_depth=current_depth + 1)  # partitions[1] == right
        return node

    def fit(self, X, y):
        """
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        """
        y = np.reshape (y, (y.shape[0], 1))
        dataset = np.concatenate ((X, y), axis=1)  # rightmost column of 'dataset' is 'y'
        self.root = self.build_tree (dataset, current_depth=0)

    def tree_traverse_to_predict(self, node, X):
        if X[node['splitting_variable']] <= node['splitting_threshold']:
            if isinstance (node['left'], dict):
                return self.tree_traverse_to_predict (node['left'], X)
            else:
                return node['left']
        else:
            if isinstance (node['right'], dict):
                return self.tree_traverse_to_predict (node['right'], X)
            else:
                return node['right']

    def predict(self, X):
        """
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        """
        predictions = list ()
        for row in X:
            predictions.append (self.tree_traverse_to_predict (self.root, row))
        return np.array (predictions)

    def get_model_string(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open (file_name, 'w') as fp:
            json.dump (model_dict, fp)


# For test
if __name__ == '__main__':
    for i in range (3):
        x_train = np.genfromtxt ("Test_data" + os.sep + "x_" + str (i) + ".csv", delimiter=",")
        y_train = np.genfromtxt ("Test_data" + os.sep + "y_" + str (i) + ".csv", delimiter=",")

        for j in range (2):
            tree = MyDecisionTreeRegressor (max_depth=5, min_samples_split=j + 2)
            tree.fit (x_train, y_train)

            model_string = tree.get_model_string ()

            with open ("Test_data" + os.sep + "decision_tree_" + str (i) + "_" + str (j) + ".json", 'r') as fp:
                test_model_string = json.load (fp)

            # with open (os.path.join ('save', 'decision_tree_gen_' + str (i) + "_" + str (j) + '.json'), 'w') as outfile:
            #     json.dump (model_string, outfile)

            print (operator.eq (model_string, test_model_string))

            y_pred = tree.predict (x_train)

            y_test_pred = np.genfromtxt (
                "Test_data" + os.sep + "y_pred_decision_tree_" + str (i) + "_" + str (j) + ".csv", delimiter=",")
            print (np.square (y_pred - y_test_pred).mean () <= 10 ** -10)
