import pickle
import pandas as pd

from eval.wrapper import BaseWrapper


class DummiesWrapper(BaseWrapper):
    WRAPPER_NAME = 'dummies'

    def __init__(self, params=None, model_file=None):
        super(DummiesWrapper, self).__init__(params, model_file)

    def apply_encoding(self, cat_num, val):
        if val not in self.encoding:
            self.max_val[cat_num] += 1
            self.encoding[cat_num][val] = self.max_val[cat_num]
        return self.encoding[cat_num][val]

    def save_model(self, fname, format="cbm", export_parameters=None):
        super(DummiesWrapper, self).save_model(fname, format, export_parameters)
        wrapper_name = self.create_wrapper_name(fname, self.WRAPPER_NAME)
        with open(wrapper_name, 'wb') as file_to:
            pickle.dump((self.encoding, self.cat_nums, self.max_val), file_to)

    def load_model(self, fname, format='catboost'):
        super(DummiesWrapper, self).load_model(fname, format)
        wrapper_name = self.create_wrapper_name(fname, self.WRAPPER_NAME)
        with open(wrapper_name, 'rb') as file_from:
            self.encoding, self.cat_nums, self.max_val = pickle.load(file_from)

    def handle_test_matrix(self, X, label):
        for num in self.cat_nums:
            col = pd.Series(X[:, num]).astype('category')
            col.apply(lambda val: self.apply_encoding(num, val))
            X[:, num] = col
        return X

    def handle_learn_matrix(self, X, label):
        self.encoding = {}
        for num in self.cat_nums:
            col = pd.Series(X[:, num]).astype('category')
            self.encoding[num] = dict(enumerate(col.cat.categories))
            X[:, num] = col.cat.codes
        self.max_val = {}
        for cat_num, mapping in self.encoding.iteritems():
            self.max_val[cat_num] = max(mapping.keys())
        return X, label
