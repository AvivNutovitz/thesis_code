import pandas as pd
import numpy as np
import shap
from scipy.sparse import csr_matrix
import matplotlib.style
import matplotlib.pyplot as plt
matplotlib.style.use('classic')


class Plotter():
    def __init__(self, x_train, plot_top=20):
        # condition on x_data
        if isinstance(x_train, pd.DataFrame):
            self.x_data = x_train.values
            self.feature_names = list(x_train.columns)
        else:
            raise ValueError(f"x_train can by pandas DataFrame ONLY, "f"but passed {type(x_train)}")
        assert len(self.feature_names) == x_train.shape[1]
        self.X_train = x_train
        self.plot_top = plot_top

    def _set_number_of_features(self, contributions=None):
        if contributions is not None:
            if contributions.shape[0] >= self.plot_top:
                return self.plot_top
            else:
                return contributions.shape[0]
        else:
            if self.plot_top >= self.X_train.shape[1]:
                return self.X_train.shape[1]
            else:
                return self.plot_top

    # shap
    def plot_shap_values_linear_model(self, model):
        shap_values = shap.LinearExplainer(model, self.X_train, nsamples=self.X_train.shape[0]).shap_values(
            self.X_train)
        shap.summary_plot(shap_values, self.X_train, plot_type="bar", feature_names=self.feature_names)

    def plot_shap_values_tree_model(self, model):
        shap_values = shap.TreeExplainer(model).shap_values(self.X_train)
        shap.summary_plot(shap_values, self.X_train, plot_type="bar", feature_names=self.feature_names)

    # feature importance
    def plot_model_coef(self, model):
        plt.figure()
        number_of_features = self._set_number_of_features()
        indices = np.argsort(model.coef_[0])[0: number_of_features]
        features_to_show = self.feature_names
        plt.title("Model Feature importance/coefficient")
        plt.barh(range(number_of_features), model.coef_[0][indices], color="b", align="center")
        plt.yticks(range(number_of_features), np.array(features_to_show)[indices.astype(int)])
        plt.ylim([-1, number_of_features])
        plt.show()

    def plot_model_importance(self, model):
        plt.figure()
        number_of_features = self._set_number_of_features()
        indices = np.argsort(model.feature_importances_)[0: number_of_features]
        features_to_show = self.feature_names
        plt.title("Model Feature importance/coefficient")
        plt.barh(range(number_of_features), model.feature_importances_[indices], color="b", align="center")
        plt.yticks(range(number_of_features), np.array(features_to_show)[indices.astype(int)])
        plt.ylim([-1, number_of_features])
        plt.show()

    # doe contribution
    def plot_doe_feature_contribution(self, class_feature_contributions,  color='b'):
        plt.figure()
        contributions = np.array([np.abs(values) for key, values in class_feature_contributions.items()])
        # Contributions = np.abs(pd.DataFrame(class_feature_contributions[class_index]).values[0])
        number_of_features = self._set_number_of_features(contributions)
        indices = np.argsort(contributions)[0: number_of_features]
        features_to_show = list(class_feature_contributions.keys())
        plt.title(f"Feature Contribution Based on Factorial Design")
        plt.barh(range(number_of_features), contributions[indices], color=color, align="center")
        plt.yticks(range(number_of_features), np.array(features_to_show)[indices.astype(int)])
        plt.ylim([-1, number_of_features])
        plt.show()