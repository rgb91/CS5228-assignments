import numpy as np
import os
import json
import operator


class MyDecisionTreeRegressor ():
    def __init__(self, max_depth=5, min_samples_split=1):
        '''
        Initialization
        :param max_depth: type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split: type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def treeGroups(self, attNo, val, X, Y):

        toReturn = {"splitting_variable": attNo, "splitting_threshold": val}
        valuesInRightClass = Y * (X[:, attNo] > val)
        valuesInLeftClass = Y * (X[:, attNo] <= val)
        lX = X[np.nonzero (X[:, attNo] <= val)]
        lY = Y[np.nonzero (X[:, attNo] <= val)]
        rX = X[np.nonzero (X[:, attNo] > val)]
        rY = Y[np.nonzero (X[:, attNo] > val)]
        toReturn["left"] = np.nanmean (lY)  # np.sum(valuesInLeftClass)/len(lY)
        toReturn["right"] = np.nanmean (rY)  # np.sum(valuesInRightClass)/len(rY)
        pred = ((toReturn["left"] * (X[:, attNo] <= val)) + (toReturn["right"] * (X[:, attNo] > val)))

        return pred, [toReturn, lX, lY, rX, rY]

    def getError(self, pred, y):
        # print (pred)
        return np.sum ((pred - y) ** 2)

    def getJS(self, X, y, pastInd):

        minError = 1000000
        minErrorConds = [{}, [], [], [], []]

        for j in range (0, X.shape[1]):

            temp = np.sort (X[:, j])

            val = temp
            for v in val:
                tempPredictions, conds = self.treeGroups (j, v, X, y)
                error = self.getError (tempPredictions, y)
                # print (j,v,error)
                if error < minError:
                    minError = error
                    minErrorConds = conds
        dt, lX, lY, rX, rY = minErrorConds
        return dt, lX, lY, rX, rY

    def buildTree(self, X, y, n, pastInd):

        if n < 1 or len (X) < self.min_samples_split:
            # if n<1:
            #     print (X)
            return np.mean (y)
        cur, lX, lY, rX, rY = self.getJS (X, y, pastInd)

        cur['left'] = self.buildTree (lX, lY, n - 1, pastInd)
        cur['right'] = self.buildTree (rX, rY, n - 1, pastInd)
        return cur

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''
        self.root = self.buildTree (X, y, self.max_depth, [])

        pass

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        preds = []
        for x in X:
            cur = self.root
            while (True):
                if isinstance (cur, dict):
                    if x[cur['splitting_variable']] > cur['splitting_threshold']:
                        cur = cur['right']
                    else:
                        cur = cur['left']
                else:
                    preds.append (cur)
                    break
        return np.asarray (preds)
        pass

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

            print (operator.eq (model_string, test_model_string))

            y_pred = tree.predict (x_train)

            y_test_pred = np.genfromtxt (
                "Test_data" + os.sep + "y_pred_decision_tree_" + str (i) + "_" + str (j) + ".csv", delimiter=",")

            print (np.square (y_pred - y_test_pred).mean () <= 10 ** -10)
