import itertools
from mlxtend.feature_selection import SequentialFeatureSelector
from scipy.sparse import csr_matrix

from design_creator import DesignCreator
from data_modifier import DataModifier
from predictor import Predictor
from validator import Validator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet, LinearRegression
from doe_xai_utils import *
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


class DoeXai:

    def __init__(self, x_data, y_data, model, feature_names=None, design_file_name=None, verbose=0, model_name=None):
        # condition on x_data
        if isinstance(x_data, pd.DataFrame):
            self.x_original_data = x_data.values
            self.feature_names = list(x_data.columns)
        elif isinstance(x_data, csr_matrix):
            self.x_original_data = x_data.toarray()
            if feature_names:
                self.feature_names = feature_names
            else:
                raise Exception("Must pass feature_names if x_data is csr_matrix")
        elif isinstance(x_data, np.ndarray):
            self.x_original_data = x_data
            if feature_names:
                self.feature_names = feature_names
            else:
                raise Exception("Must pass feature_names if x_data is np.ndarray")
        else:
            raise ValueError(f"x_data can by pandas DataFrame or numpy ndarray or scipy.sparse csr_matrix ONLY, "
                             f"but passed {type(x_data)}")
        # condition on y_data
        if isinstance(y_data, pd.DataFrame):
            self.y_original_data = y_data.values
        elif isinstance(y_data, np.ndarray):
            self.y_original_data = y_data
        elif isinstance(y_data, pd.Series):
            self.y_original_data = y_data.reset_index(drop=True)
        else:
            raise ValueError(f"y_data can by pandas DataFrame or Series or numpy ndarray ONLY, but passed {type(y_data)}")
        self.model = model
        if design_file_name:
            self.dc = DesignCreator(feature_matrix=None, file_name=design_file_name)
        else:
            self.dc = DesignCreator(feature_matrix=self.x_original_data)
        self.verbose = verbose
        self.model_name = model_name

        reference_values = [row.mean() for row in self.x_original_data.T]
        lists_of_designs, list_of_all_positions_per_design = self.dc.get_lists_of_design_from_df_for_tabluar_data(
            reference_values)

        dm = DataModifier(self.x_original_data, lists_of_designs, list_of_all_positions_per_design, len(reference_values))
        self.zeds_df, data_modified_list = dm.set_tabular_data_for_prediction()

        p = Predictor(data_modified_list, self.y_original_data, self.model, self.model_name)
        self.all_predictions_all_targets, self.all_predictions_df = p.create_tabular_gs_df()

    def find_feature_contribution(self, user_list=None, run_fffs=False, only_orig_features=False):
        self.y = self.all_predictions_df.mean(axis=1)
        self.x = self._get_x_for_feature_contribution(user_list, only_orig_features)
        self.m, self.selected_features_x = self._fit_linear_approximation(run_fffs)
        return self._create_contribution()

    def _create_contribution(self):
        contributions = {}
        for index, col in enumerate(self.selected_features_x):
            contributions[col] = np.abs(self.m.coef_[index])
        return contributions

    def _fit_linear_approximation(self, run_fffs):
        selected_features_x = list(self.x.columns)
        if run_fffs:
            feature_selector = SequentialFeatureSelector(LinearRegression(normalize=True),
                                                         k_features=max(int(np.sqrt(self.x.shape[1])), self.zeds_df.shape[1]),
                                                         forward=True,
                                                         verbose=2,
                                                         cv=5,
                                                         n_jobs=-1,
                                                         scoring='r2')

            features = feature_selector.fit(self.x, self.y)
            selected_columns = list(features.k_feature_names_)
            selected_columns.extend([list(self.x.columns)[i] for i in list(self.zeds_df.columns.astype(int))])
            selected_features_x = pd.DataFrame(self.x)[set(selected_columns)]
            m = self.get_best_linear_model(selected_features_x, self.y)
            m.fit(selected_features_x, self.y)
        else:
            m = self.get_best_linear_model(self.x, self.y)
            m.fit(self.x, self.y)

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

    @staticmethod
    def get_best_linear_model(X, y):
        model = ElasticNet(max_iter=1000) if X.shape[0] > X.shape[1] else ElasticNet(max_iter=1000, dual=True)
        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define grid
        grid = dict()
        grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0]
        grid['l1_ratio'] = [(i+1)/100 for i in range(0, 10)]
        # define search
        search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, verbose=0)
        # perform the search
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = search.fit(X, y)
        if max(results.best_estimator_.coef_) == 0:
            return LinearRegression(max_iter=1000) if X.shape[0] > X.shape[1] else LinearRegression(max_iter=1000, dual=True)
        else:
            return results.best_estimator_

