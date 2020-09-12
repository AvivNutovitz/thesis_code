import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import copy


class ContributorCalculator:
    def __init__(self, zeds_df, all_predictions_df, all_predictions_all_targets, verbose):
        self.zeds_df = zeds_df
        self.all_predictions_df = all_predictions_df
        self.gs_means = self.all_predictions_df.mean(axis=1)
        self.all_predictions_all_targets = all_predictions_all_targets
        self.all_predictions_all_targets_mean_df = pd.DataFrame(
            [all_predictions.mean() for all_predictions in self.all_predictions_all_targets])
        self.verbose = verbose

    def create_tabular_class_contributions(self):
        feature_contributions_start_time = datetime.now()
        class_feature_contributions = defaultdict(list)
        for gs_column in list(self.all_predictions_all_targets_mean_df.columns):
            # every prediction class
            if self.verbose > 0:
                print(f"start new columns: {gs_column}")
            class_feature_contributions[gs_column].append(
                self._calc_contribution(np.array(self.all_predictions_all_targets_mean_df[gs_column])))
        feature_contributions_end_time = datetime.now()
        if self.verbose > 0:
            print("finish feature contribution, total time {}".format(
                feature_contributions_end_time - feature_contributions_start_time))
        return class_feature_contributions

    def create_tabular_global_contributions(self):
        feature_contributions_start_time = datetime.now()
        global_feature_contributions = defaultdict(list)
        # every feature
        global_feature_contributions['global'].append(self._calc_contribution(np.array(self.gs_means)))
        feature_contributions_end_time = datetime.now()
        if self.verbose > 0:
            print("finish feature contribution, total time {}".format(
                feature_contributions_end_time - feature_contributions_start_time))
        return global_feature_contributions

    def _calc_contribution(self, class_prediction_column):
        return_list = []
        for zeds_col in self.zeds_df.columns:
            coalitions = np.array(self.zeds_df[zeds_col])
            #             class_feature_contributions[gs_column].append(np.sum(class_prediction_column*(b))/(np.sum(b)) - np.sum(class_prediction_column*(b-1))/(np.sum(b-1)))
            #             return_list.append((np.sum(class_prediction_column*coalitions)/np.sum(coalitions)) - np.mean(class_prediction_column))
            return_list.append((np.sum(class_prediction_column * coalitions) / np.sum(coalitions)) - (
                        np.sum(class_prediction_column * (coalitions - 1)) / np.sum(coalitions - 1)))
        return return_list

    def get_feature_contribution_and_coalition_sizes(self, input_file_name_gs, input_file_name_zeds):
        ### make better
        gs = pd.read_csv(input_file_name_gs)
        zeds = pd.read_csv(input_file_name_zeds)
        gs_means = list(gs.T.mean())
        feature_index_contributions = {}
        feature_index_coalition_sizes = {}
        for feature_i in range(zeds.shape[1]):
            # find pairs
            list_of_pairs_indexes = self.find_pairs_for_feature_i(zeds, feature_i)

            # get contribution and coalition sizes
            contributions, coalition_sizes = self.get_abs_contribution_from_pairs_and_coalition_sizes(gs_means, zeds,
                                                                                                 list_of_pairs_indexes)
            feature_index_contributions[feature_i] = contributions
            feature_index_coalition_sizes[feature_i] = coalition_sizes

        return pd.DataFrame(feature_index_contributions), pd.DataFrame(feature_index_coalition_sizes)

    @staticmethod
    def find_pairs_for_feature_i(zeds, feature_i):
        zeds_copy = copy.deepcopy(zeds)
        map_dict = defaultdict(list)
        for index, zed in enumerate(zeds_copy.iterrows()):
            l = list(zed[1])
            del l[feature_i:feature_i + 1]
            map_dict[repr(l)].append(index)
        return [values for keys, values in map_dict.items()]

    @staticmethod
    def get_abs_contribution_from_pairs_and_coalition_sizes(gs, zeds, list_of_pairs_indexes):
        contributions = []
        coalition_sizes = []
        for index_pair in list_of_pairs_indexes:
            coalition_size = min(sum(zeds.iloc[index_pair[0], :]), sum(zeds.iloc[index_pair[1], :]))
            coalition_sizes.append(coalition_size)
            contribution = np.abs(gs[index_pair[0]] - gs[index_pair[1]])
            contributions.append(contribution)
        return contributions, coalition_sizes
