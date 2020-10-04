# --- Imports
from doe_utils import load_data
from doe_xai import DoeXai
from plotter import Plotter

# from .fake_job_posting_example import create_fake_job_posting_data_and_tv
# from .hotel_bookings_example import create_hotel_booking_data
from test_examples import create_fake_job_posting_data_and_tv

tv_train_reviews, y_train, tv_test_reviews, y_test, tv= create_fake_job_posting_data_and_tv()
print(tv_train_reviews.shape)