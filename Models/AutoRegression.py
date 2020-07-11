import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model import LinearRegression

class AutoRegression(BaseEstimator, RegressorMixin):
    # Constructor
    def __init__(self, max_shift = None, notSubFeatures = []):
        self.max_shift = max_shift
        self.notSubFeatures = notSubFeatures
        self.coef_tot = []
        self.coef_subs = []
        self.tot_lr = None
        self.subs_lr = []
        self.subs_predict = None
        self.tot_predict = None

    # Check data validation
    def validate_data(self, X, y):
        assert self.same_length(X.index, y.index), "The X and Y have different length"
        assert self.all_shifts(X), "Not all the relevant shifts in the features"
        return True

    # Validation check - same length of X and y
    def same_length(self, X_index, y_index):
        if X_index.min() != y_index.min() or X_index.max() != y_index.max() or len(X_index) != len(y_index):
            return False
        return True

    # Validation check - all the relevant shift (from 0 included)
    def all_shifts(self, X):
        unique_features_num = len(set([feature.split('_')[0] for feature in X.columns if feature not in self.notSubFeatures]))
        for num in range(self.max_shift+1):
            if len(self.get_shift(X[X.columns.difference(self.notSubFeatures)], num)) != unique_features_num:
                return False
        return True

    #
    def get_shift(self, X, num):
        # Changed to endswith(Might be shift. in the middle - Just in case).
        return [feature for feature in X.columns if feature.endswith(f'shift.{num}') and feature not in self.notSubFeatures]

    def remove_shift(self, feature):
        feature = feature[:-7]

    def get_max_shift(self, X):
        components = {}
        for feature in X[X.columns.difference(self.notSubFeatures)]:
            shift_num = get_shift(feature)
            component_name = self.remove_shift(feature)
            if not component_name in components:
                components[component_name] = shift_num
            else:
                components[component_name] = max(shift_num, components[component_name])
        min_shift_num = min(components.values())
        max_shift_num = max(components.values())
        assert min_shift_num == max_shift_num, "The components don't have the same shifts."
        return max_shift_num

    def clean_irrelevant_shifts(self,X):
        # Delete irrelevant shifts - features from DB with highest shift
        shift_str = '.shift.'
        for feature in X.columns:
            ind_shift = feature.find(shift_str, 0) + len(shift_str)
            if feature not in self.notSubFeatures and self.max_shift < int(feature[ind_shift:]):
                X.drop([feature], axis=1, inplace=True)

    def clean_notsub_shifts(self,X):
        # Delete irrelevant shifts - notsub
        shift_str = '.shift.'
        for feature in self.notSubFeatures:
            ind_shift = feature.find(shift_str, 0) + len(shift_str)
            for shift in range(self.max_shift + 1):
                if shift != int(feature[ind_shift:]) and feature[:ind_shift] + str(shift) in X.columns:
                    X.drop([feature[:ind_shift] + str(shift)], axis=1, inplace=True)

    def fit(self, X, y):
        if self.max_shift == None:
            self.max_shift = self.get_max_shift(X)
        self.clean_irrelevant_shifts(X)
        self.clean_notsub_shifts(X)
        if self.validate_data(X, y):
            self.features = []
            self.coef_subs = []
            self.subs_lr = []
            self.tot_lr_coef(X, y)
            self.subs_lr_coef(X)
            self.coef_ = np.hstack((self.coef_tot, np.hstack(self.coef_subs)))

    def get_features(self):
        return self.features

    def replace_shift(self, feature, number):
        pos = feature.index(".shift.")
        feature = feature[0:pos] + f".shift.{number}"
        return feature

    def tot_lr_coef(self, X, y):
        if self.notSubFeatures is not None:
            list_shift_zero = self.get_shift(X, 0) + self.notSubFeatures
        else:
            list_shift_zero = self.get_shift(X, 0)
        tot_lr = LinearRegression()
        tot_lr.fit(X[list_shift_zero], y)
        self.features = list_shift_zero
        self.coef_tot = tot_lr.coef_
        self.tot_lr = tot_lr

    def subs_lr_coef(self, X):
        sub_classes = self.shift_features(X)
        for sub in sub_classes:
            lr = LinearRegression()
            target = self.replace_shift(sub[0], 0)
            lr.fit(X[sub], X[target])
            self.coef_subs.append(lr.coef_)
            self.features += sub
            self.subs_lr.append(lr)

    def shift_features(self, X):
        unique_features = [feature.split('_')[0] for feature in X.columns if get_shift(feature) == 0 and (feature not in self.notSubFeatures)]
        shifts = np.arange(1, self.max_shift + 1)
        shift_features = []
        for unique_feature in unique_features:
            shift_features.append([feature for feature in X.columns
                                   if ((unique_feature == feature.split('_')[0]) and
                                       (get_shift(feature) in shifts))])
        return shift_features

    def predict(self, X):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X).transpose()
        self.clean_irrelevant_shifts(X)
        self.clean_notsub_shifts(X)
        assert self.all_shifts(X), "Not all the relevant shifts in the features"
        return self.predict_sub_classes(X)

    def predict_sub_classes(self, X):
        predict_columns = [feature.split('.')[0] + '_Pred' for feature in X.columns if get_shift(feature) == 0 and feature not in self.notSubFeatures]
        df_subs_pred = pd.DataFrame(index=X.index, columns=predict_columns)
        sub_classes = self.shift_features(X)
        for ind, sub in enumerate(sub_classes):
            df_subs_pred[sub[0].split('.')[0] + '_Pred'] = self.subs_lr[ind].predict(X[sub])
        self.subs_predict = df_subs_pred
        df_subs_pred = pd.concat([df_subs_pred,X[self.notSubFeatures]],axis=1)
        self.tot_predict = self.tot_lr.predict(df_subs_pred)
        return self.tot_predict