# --- imports
from test_examples import create_hotel_booking_data
from doe_xai import DoeXai
from plotter import Plotter

# thanks to - https://www.kaggle.com/marcuswingen/eda-of-bookings-and-ml-to-predict-cancelations for the preprocess and model training

# --- Other imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from numpy import random
import shap
seed = 42

if __name__ == '__main__':
    # --- Data Prepossess
    X_train, y_train, X_test, y_test = create_hotel_booking_data()

    # --- Model Training
    model = LogisticRegression(random_state=random.seed(seed), n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)
    score = roc_auc_score(y_test, [p[1] for p in preds])
    print(f"auc_score: {round(score, 4)}")
    print("=" * 80)
    print(classification_report(y_test, model.predict(X_test)))

    # --- SHAP
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    print(X_train.columns)
    # --- DOE
    dx = DoeXai(x_data=X_train, y_data=y_train, model=model, feature_names=X_train.columns)
    # features:
    """['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
       'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
       'babies', 'is_repeated_guest', 'previous_cancellations',
       'previous_bookings_not_canceled', 'agent', 'company',
       'required_car_parking_spaces', 'total_of_special_requests', 'adr',
       'hotel_Resort Hotel', 'arrival_date_month_August',
       'arrival_date_month_July', 'meal_BB', 'meal_FB', 'meal_HB',
       'market_segment_Complementary', 'market_segment_Corporate',
       'market_segment_Direct', 'market_segment_Groups',
       'market_segment_Offline TA/TO', 'market_segment_Online TA',
       'distribution_channel_Corporate', 'distribution_channel_Direct',
       'distribution_channel_TA/TO', 'reserved_room_type_A',
       'reserved_room_type_C', 'reserved_room_type_D', 'reserved_room_type_E',
       'reserved_room_type_F', 'reserved_room_type_G', 'reserved_room_type_H',
       'reserved_room_type_L', 'deposit_type_No Deposit',
       'customer_type_Contract', 'customer_type_Group',
       'customer_type_Transient', 'customer_type_Transient-Party']"""
    features_to_test = [['company', 'agent'], ['company', 'adults', 'children']]
    cont = dx.find_feature_contribution(user_list=features_to_test)

    # --- Plot
    p = Plotter(X_train)
    p.plot_doe_feature_contribution(cont)