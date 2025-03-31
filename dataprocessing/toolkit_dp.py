"""
1 doc to rule them all - for A5 Group 3 Data Processing
"""

import numpy as np
import pandas as pd
import sqlite3
import os


class DatasetProcessor:
    def __init__(
        self, radar_path: str = "datasets/radar_manchester_2021.csv", met_path: str = "datasets/all_met.csv",
        plan_path: str = "datasets/plan_manchester_2021.csv", adsb_path: str = "datasets/2021_all_manchester_data_ADSB",
        features_path: str = "datasets/man_features", features_month_sep: bool = True
    ):
        self.radar_path = radar_path
        self.met_path = met_path
        self.plan_path = plan_path
        self.adsb_path = adsb_path
        self.features_path = features_path
        self.features_month_sep = features_month_sep
        self.status = {
            "load": {
                "met": False,
                "plan": False,
                "radar": False,
                "features": False,
                "adsb": False,
            },
            "clean": {
                "radar": False,
                "adsb": False,
                "met": False,
                "plan": False,
                "features": False,
            },
            "generate": {
                "tt": False,
                "plan_features": False,
            },
            "final": {
                "merge_datasets": False,
                "create_train_test": False
            }
        }  # to make it more kid-friendly

    def print_status(self):
        print("=========================")
        print("STATUS".center(25))  # centres the text - had no idea it existed
        print("=========================")

        for section, tasks in self.status.items():
            print(f"\n{section.upper()}".center(25))
            print("-------------------------")
            for task, completed in tasks.items():
                status_symbol = "✓" if completed else "✗"  # Unicode for the win
                print(f"{task:<20}: [{status_symbol}]")  # incase it bleeds out
        print("-------------------------")

    def load_adsb(self, clean_path: str = None):
        if self.status["load"]["adsb"]:
            raise Exception("Has already been loaded")

        if clean_path is not None:
            self.adsb_df = pd.read_csv(self.adsb_path)
        else:
            self.adsb_path = clean_path
            self.adsb_df = pd.read_csv(self.adsb_path)
            self.status["clean"]["adsb"] = True  # as ADSB takes a while to process - so no point operating if exists

        self.status["load"]["adsb"] = True

    def load_met(self):
        if self.status["load"]["met"]:
            raise Exception("Has already been loaded")

        self.met_df = pd.read_csv(self.met_path)
        self.status["load"]["met"] = True

    def load_plan(self):
        if self.status["load"]["plan"]:
            raise Exception("Has already been loaded")

        self.plan_df = pd.read_csv(self.plan_path)
        self.status["load"]["plan"] = True

    def load_radar(self):
        if self.status["load"]["radar"]:
            raise Exception("Has already been loaded")

        self.radar_df = pd.read_csv(self.radar_path)
        self.status["load"]["radar"] = True

    def load_features(self):
        if self.status["load"]["features"]:
            raise Exception("Has already been loaded")

        if not self.features_month_sep:
            self.features_df = pd.read_csv(self.features_path)
        else:
            file_names = [f"{self.features_path}/{file}" for file in os.listdir(self.features_path)]
            self.features_df = pd.concat([pd.read_csv(path) for path in file_names]).reset_index(drop=True)
        self.status["load"]["features"] = True

    def load_all(self):
        if True in self.status["load"].values():
            raise Exception("Either some or all of the datasets have already been loaded")

        self.adsb_df = self.load_adsb()
        self.met_df = self.load_met()
        self.plan_df = self.load_plan()
        self.radar_df = self.load_radar()
        self.features_df = self.load_features()

        for func in self.status["load"]:
            self.status["load"][func] = True

    def clean_radar(self):
        if not self.status["load"]["radar"]:
            raise Exception("The dataset has not yet been loaded")

        self.radar_df = self.radar_df[["mode_s_hex", "datatime_utc", "climb_rate_calc"]].dropna(
            subset=["mode_s_hex", "climb_rate_calc"]
        )  # as are essential cols, so if they don't exist then drop

        self.radar_df['datetime_utc'] = pd.to_datetime(self.radar_df['datetime_utc'])  # self explainatory
        self.radar_df['unix_time'] = self.radar_df['datetime_utc'].astype(int) / 10**9  # cvt unixtime

        self.radar_df = self.radar_df.groupby('mode_s_hex').apply(
            lambda x: x.sort_values('unix_time')
        ).reset_index(drop=True)  # groups them so would be easier to calc tt later
        self.radar_df = self.radar_df[self.radar_df['climb_rate_calc'] != 0]  # removes points when climb = 0

        self.status["clean"]["radar"] = True

    def clean_adsb(self):
        if not self.status["load"]["adsb"]:
            raise Exception("The dataset has not yet been loaded")

        self.status["clean"]["adsb"] = True

    def clean_met(self):
        if not self.status["load"]["met"]:
            raise Exception("The dataset has not yet been loaded")

        self.met_df.drop(columns=['Unnamed: 0.1','Unnamed: 0'], axis=1, inplace=True)  # represent ind in 2 cols
        self.met_df = self.met_df.dropna(axis=1, how='all')  # 0 entries
        self.met_df.drop(columns='wind_rose_dir', axis=1, inplace=True)  # another col represents this data, but better
        self.df_met['datetime'] = pd.to_datetime(self.df_met['datetime'])  # makes sense man

        self.df_met['base_viz_m'] = self.df_met['base_viz_m'].replace('> 10k','9999')  # makes everything a float
        self.df_met['base_viz_m'] = self.df_met['base_viz_m'].astype(float)

        self.df_met = self.df_met.drop(self.df_met[self.df_met['windspeed_kts'] >= 50].index)  # >= 50 are outliers

        self.status["clean"]["met"] = True

    def clean_plan(self):
        if not self.status["load"]["plan"]:
            raise Exception("The dataset has not yet been loaded")

        self.status["clean"]["plan"] = True

    def clean_features(self):
        if not self.status["load"]["features"]:
            raise Exception("The dataset has not yet been loaded")

        # doesn't need cleaning as should be cleaned already - created as might add some pre-processing if needed

        self.status["clean"]["features"] = True

    def clean_all(self):
        if False in self.status["load"].values():
            raise Exception("Either some or all of the datasets have not been loaded")

        self.clean_radar()
        self.clean_adsb()
        self.clean_features()
        self.clean_plan()
        self.clean_met()

        for func in self.status["clean"]:
            self.status["clean"][func] = True

    def generate_tt(self):
        if not self.status["clean"]["radar"] or not self.status["clean"]["adsb"]:
            raise Exception("Either or both the datasets - ADSB and Radar have not been loaded/cleaned")

        # radar
        self.radar_df = self.radar_df[(self.radar_df['time_diff'] <= 86400) & (self.radar_df['time_diff'] >= 900)]
        self.radar_df = self.radar_df.sort_values(by='time_diff')  # does the calc

        # adsb :(

        # merge

        self.status["generate"]["turnaround_time"] = True

    def generate_plan_features(self):
        if not self.status["clean"]["features"]:
            raise Exception("Feature dataset has either not been cleaner or not even loaded")

        self.status["generate"]["plan_features"] = True

    def merge_datasets(self):
        if False in self.status["generate"].values():
            raise Exception("Not all necessary data has been processed in the datasets")

        self.status["final"]["merge_datasets"] = True

    def create_train_test(self):
        if not self.status["final"]["merge_datasets"]:
            raise Exception("Dataset has not yet been merged")

        self.status["final"]["create_train_test"] = True


if __name__ == "__main__":
    processor = DatasetProcessor()
    processor.print_status()
