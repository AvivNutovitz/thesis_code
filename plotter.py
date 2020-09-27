import pandas as pd
import numpy as np
import shap
from scipy.sparse import csr_matrix
import matplotlib.style
import matplotlib.pyplot as plt
matplotlib.style.use('classic')


class Plotter():
    def __init__(self, x_train, feature_names):
        # condition on x_data
        if isinstance(x_train, pd.DataFrame):
            self.x_data = x_train.values
            self.feature_names = list(x_train.columns)
        elif isinstance(x_train, csr_matrix):
            self.x_data = x_train.toarray()
            if feature_names:
                self.feature_names = feature_names
            else:
                raise Exception("Must pass feature_names if x_train is csr_matrix")
        elif isinstance(x_train, np.ndarray):
            self.x_data = x_train
            if feature_names:
                self.feature_names = feature_names
            else:
                raise Exception("Must pass feature_names if x_train is np.ndarray")
        else:
            raise ValueError(f"x_train can by pandas DataFrame or numpy ndarray or scipy.sparse csr_matrix ONLY, "
                             f"but passed {type(x_train)}")
        assert len(feature_names) == x_train.shape[1]
        self.X_train = x_train
        self.feature_names = feature_names

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
        indices = np.argsort(model.coef_[0])
        features_to_show = self.feature_names
        plt.title("Model Feature importance/coefficient")
        plt.barh(range(self.X_train.shape[1]), model.coef_[0][indices], color="b", align="center")
        plt.yticks(range(self.X_train.shape[1]), np.array(features_to_show)[indices.astype(int)])
        plt.ylim([-1, self.X_train.shape[1]])
        plt.show()

    def plot_model_importance(self, model):
        plt.figure()
        indices = np.argsort(model.feature_importances_)
        features_to_show = self.feature_names
        plt.title("Model Feature importance/coefficient")
        plt.barh(range(self.X_train.shape[1]), model.feature_importances_[indices], color="b", align="center")
        plt.yticks(range(self.X_train.shape[1]), np.array(features_to_show)[indices.astype(int)])
        plt.ylim([-1, self.X_train.shape[1]])
        plt.show()

    # doe contribution
    def plot_doe_feature_contribution(self, class_feature_contributions,  color='b'):
        plt.figure()
        contributions = np.array([np.abs(values) for key, values in class_feature_contributions.items()])
        # Contributions = np.abs(pd.DataFrame(class_feature_contributions[class_index]).values[0])
        indices = np.argsort(contributions)
        features_to_show = list(class_feature_contributions.keys())
        plt.title(f"Feature contribution Based on 2 Factor Design Features Contribution")
        plt.barh(range(len(features_to_show)), contributions[indices], color=color, align="center")
        plt.yticks(range(len(features_to_show)), np.array(features_to_show)[indices.astype(int)])
        plt.ylim([-1, len(features_to_show)])
        plt.show()