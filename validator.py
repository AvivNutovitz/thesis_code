import pandas as pd
import numpy as np
import scipy.stats as stats


class Validator:
    def __init__(self, feature_names, shap_values, class_feature_contributions):
        self.feature_names = feature_names
        self.max_display = len(feature_names)
        self.shap_values = shap_values
        self.class_feature_contributions = class_feature_contributions

    def get_all_classes_features_order_from_shap(self):
        all_classes_features_order = {}
        if len(self.shap_values) > 2:
            for prediction_cls in range(0, len(self.shap_values)):
                class_feature_order = np.argsort(np.sum(np.abs(self.shap_values[prediction_cls]), axis=0))
                class_feature_order = class_feature_order[-min(self.max_display, len(class_feature_order)):]
                all_classes_features_order[prediction_cls] = class_feature_order
            return all_classes_features_order

    def get_all_class_features_order_from_contributions(self):
        all_classes_features_order = {}
        for class_index, value in self.class_feature_contributions.items():
            indices = np.argsort(np.abs(pd.DataFrame(self.class_feature_contributions[class_index]).values[0]))
            all_classes_features_order[class_index] = indices
        return all_classes_features_order

    def get_kendalltau_st_and_pv_all_classes(self):
        shap_features_order = self.get_all_classes_features_order_from_shap()
        contributions_features_order = self.get_all_class_features_order_from_contributions()
        results = []
        for class_index in contributions_features_order.keys():
            tau, p_value = self._run_kendalltau_test(shap_features_order, contributions_features_order, class_index)
            results.append((tau, p_value))
        return results

    def get_kendalltau_st_and_pv_global(self, class_feature_contributions_global):
        shap_feature_order = np.argsort(np.sum(np.mean(np.abs(self.shap_values), axis=1), axis=0))
        shap_feature_order = shap_feature_order[-min(self.max_display, len(shap_feature_order)):]
        contributions_features_order = np.argsort(
            np.abs(pd.DataFrame(class_feature_contributions_global['global']).values[0]))
        return stats.kendalltau(shap_feature_order, contributions_features_order)

    def _run_kendalltau_test(self, d1, d2, key):
        return stats.kendalltau(d1[key], d2[key])