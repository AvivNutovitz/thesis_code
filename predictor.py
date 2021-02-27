import pandas as pd
import numpy as np


class Predictor:
    def __init__(self, data_modified_list, target_df, model, model_name=None):
        self.data_modified_list = data_modified_list
        self.target_df = target_df
        self.model = model
        self.model_name = model_name

    def create_tabular_gs_df(self):
        all_predictions_one_target = []
        all_predictions_all_targets = []
        for data_modified in self.data_modified_list:
            # keras models
            if self.model_name == 'CNN' or self.model_name == 'DNN':
                # CNN
                case_predictions = pd.DataFrame(self.model.predict(data_modified).astype(float))
            else:
                # other model types
                case_predictions = pd.DataFrame(self.model.predict_proba(data_modified))
            all_predictions_all_targets.append(case_predictions)
            case_predictions_one_target = []
            for v in case_predictions.iterrows():
                tmp = v[1][self.target_df[v[0]]]
                if isinstance(tmp, float):
                    case_predictions_one_target.append(tmp)
                elif isinstance(tmp, pd.Series):
                    case_predictions_one_target.append(tmp.values[0])
            all_predictions_one_target.append(case_predictions_one_target)

        return all_predictions_all_targets, pd.DataFrame(all_predictions_one_target)

    # -- for image data not in use now --
    # -----------------------------------
    def create_image_predictions(self, predictions_df):
        for data_modified in self.data_modified_list:
            predictions_df.append(pd.DataFrame(self.model.predict(np.expand_dims(data_modified, axis=0))))
        return pd.concat(predictions_df)
