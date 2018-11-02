import numpy as np
import os
import json
import operator


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


    @staticmethod
    def info_gain(dataset, partitions, y):
        left, right = np.array (partitions[0]), np.array (partitions[1])
        y = np.array (y)
        a_hat = np.mean (y)
        y_left, y_right = None, None
        if left.shape[0] > 0:
            y_left = left[:, -1]
        else:
            y_left = np.array ([])
        if right.shape[0] > 0:
            y_right = right[:, -1]
        else:
            y_right = np.array ([])
        a_hat_left, a_hat_right = 0, 0
        impurity_of_dataset = np.sum (np.square (a_hat - y)) / float (len (dataset))
        if left.shape[0] == 0:
            impurity_of_left_set = 0
        else:
            a_hat_left = np.nanmean (y_left)
            impurity_of_left_set = np.sum (np.square (a_hat_left - y_left)) / float (len (left))
        if right.shape[0] == 0:
            impurity_of_right_set = 0
        else:
            a_hat_right = np.nanmean (y_right)
            impurity_of_right_set = np.sum (np.square (a_hat_right - y_right)) / float (len (right))

        gain = impurity_of_dataset - (float (len (left)) / float (len (dataset)) * impurity_of_left_set) - (float (len (right)) / float (len (dataset)) * impurity_of_right_set)
        return gain

    @staticmethod
    def test_split(index, value, dataset):  # Creates test split while calculating best Info. Gain in Build Tree
        left, right = list (), list ()
        for row in dataset:
            if row[index] <= value:
                left.append (row)
            else:
                right.append (row)
        return left, right

    def get_split_node(self, dataset):
        y_values = list (set (row[-1] for row in dataset))
        best_index, best_value, max_info_gain, best_partitions = 9999, 9999, 0, None
        X_values = np.array(dataset)
        X_values = X_values[:, :-1]
        for i in range (len (dataset[0]) - 1):
            sorted_X = np.sort(X_values[:, i])
            # for row in dataset:
            #     partitions = self.test_split (index=i, value=row[i], dataset=dataset)
            for val in sorted_X:
                partitions = self.test_split (index=i, value=val, dataset=dataset)
                ig = self.info_gain (dataset, partitions, y_values)
                if ig > max_info_gain:
                    best_index, best_value, max_info_gain, best_partitions = i, val, ig, partitions
        return {'splitting_variable': best_index, 'splitting_threshold': best_value, 'partitions': best_partitions}

    @staticmethod
    def to_terminal(partition):
        p = np.array (partition)
        p = p[:, -1]
        return np.mean (p)

    def split(self, node_to_split, depth):
        left, right = node_to_split['partitions']
        del (node_to_split['partitions'])

        # If no left or right node exist
        if not left or not right:
            node_to_split['left'] = node_to_split['right'] = self.to_terminal (left + right)
            return

        # If max depth is reached
        if depth >= self.max_depth:
            node_to_split['left'], node_to_split['right'] = self.to_terminal (left), self.to_terminal (right)
            return

        # Process left node
        if len (left) < self.min_samples_split:
            node_to_split['left'] = self.to_terminal (left)
        else:
            node_to_split['left'] = self.get_split_node (left)
            self.split (node_to_split['left'], depth=depth + 1)

        # Process right node
        if len (right) < self.min_samples_split:
            node_to_split['right'] = self.to_terminal (right)
        else:
            node_to_split['right'] = self.get_split_node (right)
            self.split (node_to_split['right'], depth=depth + 1)

    def build_tree(self, dataset):
        """
        Build the Decision Tree
        :param X: Train feature data, type: numpy array, shape: (N, num_feature)
        :param y: Train label data, type: numpy array, shape: (N,)
        :return: tree: Fitted Decision Tree, type: dictionary
        """
        root = self.get_split_node (dataset)
        self.split (root, depth=1)
        self.root = root

    def fit(self, X, y):
        """
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        """
        y = np.reshape (y, (y.shape[0], 1))
        dataset = np.concatenate ((X, y), axis=1)  # rightmost column of 'dataset' is 'y'
        self.build_tree (dataset)

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
            print ()
