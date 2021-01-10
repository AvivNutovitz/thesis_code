import itertools
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector
from scipy.sparse import csr_matrix

from design_creator import DesignCreator
from data_modifier import DataModifier
from predictor import Predictor
from validator import Validator
from doe_utils import *


class DoeXai:

    def __init__(self, x_data, y_data, model, feature_names=None, design_file_name=None, verbose=0):
        # condition on x_data
        if isinstance(x_data, pd.DataFrame):
            self.x_data = x_data.values
            self.feature_names = list(x_data.columns)
        elif isinstance(x_data, csr_matrix):
            self.x_data = x_data.toarray()
            if feature_names:
                self.feature_names = feature_names
            else:
                raise Exception("Must pass feature_names if x_data is csr_matrix")
        elif isinstance(x_data, np.ndarray):
            self.x_data = x_data
            if feature_names:
                self.feature_names = feature_names
            else:
                raise Exception("Must pass feature_names if x_data is np.ndarray")
        else:
            raise ValueError(f"x_data can by pandas DataFrame or numpy ndarray or scipy.sparse csr_matrix ONLY, "
                             f"but passed {type(x_data)}")
        # condition on y_data
        if isinstance(y_data, pd.DataFrame):
            self.y_data = y_data.values
        elif isinstance(y_data, np.ndarray):
            self.y_data = y_data
        elif isinstance(y_data, pd.Series):
            self.y_data = y_data.reset_index(drop=True)
        else:
            raise ValueError(f"y_data can by pandas DataFrame or Series or numpy ndarray ONLY, but passed {type(y_data)}")
        self.model = model
        if design_file_name:
            self.dc = DesignCreator(feature_matrix=None, file_name=design_file_name)
        else:
            self.dc = DesignCreator(feature_matrix=self.x_data)
        self.verbose = verbose

        reference_values = [row.mean() for row in self.x_data.T]
        lists_of_designs, list_of_all_positions_per_design = self.dc.get_lists_of_design_from_df_for_tabluar_data(
            reference_values)

        dm = DataModifier(self.x_data, lists_of_designs, list_of_all_positions_per_design, len(reference_values))
        self.zeds_df, data_modified_list = dm.set_tabular_data_for_prediction()

        p = Predictor(data_modified_list, self.y_data, self.model)
        self.all_predictions_all_targets, self.all_predictions_df = p.create_tabular_gs_df()

    def find_feature_contribution(self, user_list=None, run_fffs=False, only_orig_features=False):
        y = self.all_predictions_df.mean(axis=1)
        x = self._get_x_for_feature_contribution(user_list, only_orig_features)
        m, selected_features_x = self._fit_linear_approximation(x, y, run_fffs)
        return self._create_contribution(m, selected_features_x)

    @staticmethod
    def _create_contribution(m, selected_features_x):
        contributions = {}
        for index, col in enumerate(selected_features_x):
            contributions[col] = m.coef_[index]
        return contributions

    def _fit_linear_approximation(self, x, y, run_fffs):
        m = LinearRegression(normalize=True)
        selected_features_x = list(x.columns)
        if run_fffs:
            feature_selector = SequentialFeatureSelector(LinearRegression(normalize=True),
                                                         k_features=max(int(np.sqrt(x.shape[1])), self.zeds_df.shape[1]),
                                                         forward=True,
                                                         verbose=2,
                                                         cv=5,
                                                         n_jobs=-1,
                                                         scoring='r2')

            features = feature_selector.fit(x, y)
            selected_columns = list(features.k_feature_names_)
            selected_columns.extend([list(x.columns)[i] for i in list(self.zeds_df.columns.astype(int))])
            selected_features_x = pd.DataFrame(x)[set(selected_columns)]
            m.fit(selected_features_x, y)
        else:
            m.fit(x, y)

        return m, selected_features_x

    def _get_x_for_feature_contribution(self, user_list=None, only_orig_features=False):
        x = self.zeds_df.copy()
        try:
            x.columns = self.feature_names
        except:
            pass

        if only_orig_features:
            return x

        if user_list:
            for new_feature in user_list:
                feature_name = str(new_feature[0])
                feature_value = x[new_feature[0]]
                for index, elements in enumerate(new_feature):
                    if index > 0:
                        feature_name += '-' + str(new_feature[index])
                        feature_value = feature_value * x[new_feature[index]]
                x[feature_name] = feature_value
        else:
            list_of_columns_pairs = list(itertools.combinations(x.columns, 2))
            for pair in list_of_columns_pairs:
                new_feature = str(pair[0]) + '-' + str(pair[1])
                x[new_feature] = x[pair[0]] * x[pair[1]]

        x.columns = x.columns.astype(str)
        return x

    def output_process_files(self, output_files_prefix):
        # self.df_output.to_csv(f'{output_files_prefix}_feature_contributions.csv', index=False)
        self.zeds_df.to_csv(f'{output_files_prefix}_zeds_df.csv', index=False)
        self.all_predictions_df.to_csv(f'{output_files_prefix}_gs_df.csv', index=False)

