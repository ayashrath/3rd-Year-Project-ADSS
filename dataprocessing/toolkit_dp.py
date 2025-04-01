"""
1 doc to rule them all - for A5 Group 3 Data Processing
"""

import numpy as np
import pandas as pd
import sqlite3
import os
from geopy.distance import great_circle


class DatasetProcessor:
    def __init__(
        self, radar_path: str = "datasets/radar_manchester_2021.csv", met_path: str = "datasets/all_met.csv",
        plan_path: str = "datasets/plan_manchester_2021.csv", features_path: str = "datasets/man_features",
        adsb_path: str = "datasets/2021_all_manchester_data_ADSB.csv", airports_path: str = "datasets/airports.csv",
        features_month_sep: bool = True
    ):
        self.radar_path = radar_path
        self.met_path = met_path
        self.plan_path = plan_path
        self.adsb_path = adsb_path
        self.features_path = features_path
        self.airports_path = airports_path
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
        )  # as are essential cols, so if they don"t exist then drop

        self.radar_df["datetime_utc"] = pd.to_datetime(self.radar_df["datetime_utc"])  # self explainatory
        self.radar_df["unix_time"] = self.radar_df["datetime_utc"].astype(int) / 10**9  # cvt unixtime

        self.radar_df = self.radar_df.groupby("mode_s_hex").apply(
            lambda x: x.sort_values("unix_time")
        ).reset_index(drop=True)  # groups them so would be easier to calc tt later
        self.radar_df = self.radar_df[self.radar_df["climb_rate_calc"] != 0]  # removes points when climb = 0

        self.status["clean"]["radar"] = True

    def clean_adsb(self):
        if not self.status["load"]["adsb"]:
            raise Exception("The dataset has not yet been loaded")

        self.status["clean"]["adsb"] = True

    def clean_met(self):
        if not self.status["load"]["met"]:
            raise Exception("The dataset has not yet been loaded")

        self.met_df.drop(columns=["Unnamed: 0.1","Unnamed: 0"], axis=1, inplace=True)  # represent ind in 2 cols
        self.met_df = self.met_df.dropna(axis=1, how="all")  # 0 entries
        self.met_df.drop(columns="wind_rose_dir", axis=1, inplace=True)  # another col represents this data, but better
        self.df_met["datetime"] = pd.to_datetime(self.df_met["datetime"])  # makes sense man

        self.df_met["base_viz_m"] = self.df_met["base_viz_m"].replace("> 10k","9999")  # makes everything a float
        self.df_met["base_viz_m"] = self.df_met["base_viz_m"].astype(float)

        self.df_met = self.df_met.drop(self.df_met[self.df_met["windspeed_kts"] >= 50].index)  # >= 50 are outliers

        self.status["clean"]["met"] = True

    def clean_plan(self):
        if not self.status["load"]["plan"]:
            raise Exception("The dataset has not yet been loaded")

        self.plan_df["route_grex"] = self.plan_df["route_grex"].apply(
            lambda x: f"{x}]" if pd.notna(x) and not str(x).endswith("]") else x
        )  # fix inconsistencies in grex

        self.plan_df["scs"] = self.plan_df["scs"].apply(
            lambda x: np.nan if "255" in str(x) else x
        )  # making na values into 255 (why idk)

        self.status["clean"]["plan"] = True

    def clean_features(self):
        if not self.status["load"]["features"]:
            raise Exception("The dataset has not yet been loaded")

        # doesn"t need cleaning as should be cleaned already - created as might add some pre-processing if needed

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
        self.radar_df = self.radar_df[(self.radar_df["time_diff"] <= 86400) & (self.radar_df["time_diff"] >= 900)]
        self.radar_df = self.radar_df.sort_values(by="time_diff")  # does the calc

        # adsb :(

        # merge

        self.status["generate"]["turnaround_time"] = True

    def generate_plan_features(self, country_icao: str = "EG", ):
        if not self.status["clean"]["features"]:
            raise Exception("Feature dataset has either not been cleaner or not even loaded")

        airports = sorted(set(self.plan_df["origin"]).union(set(self.plan_df["dest"])))  # makes list of all aircrafts
        uk_airports = [airport for airport in airports if airport.startswith(country_icao)]  
        # as ICAO gives EGxx for UK airport (for UK there are 3 airports that should not be considered domestic)
        # 2 in Antartica and 1 in Falkland Islands - But there are no flights from Manchester that go there :)

        # if a flight is domestic or international
        def classify_flight(row):
            if row["origin"] in uk_airports and row["dest"] in uk_airports:
                return "Domestic"
            return "International"

        self.plan_df["flight_type"] = self.plan_df.apply(classify_flight, axis=1)

        # short or long haul
        # according to dictionary of aviation (2nd edition - 2007), short haul is < 1000 km else long haul
        airports_df = pd.read_csv(self.airports_path)
        airports_df = airports_df[["latitude_deg", "longitude_deg", "icao_code"]]

        def get_airport_coordinates(icao):
            airport = airports_df[airports_df["icao_code"] == icao]
            if not airport.empty:
                return airport.iloc[0]["latitude_deg"], airport.iloc[0]["longitude_deg"]
            return None, None

        def classify_flight_distance(icao1, icao2):
            lat1, lon1 = get_airport_coordinates(icao1)
            lat2, lon2 = get_airport_coordinates(icao2)

            if lat1 is None or lat2 is None:
                return "Unknown"

            distance_km = great_circle((lat1, lon1), (lat2, lon2)).km  # calcs in km using lat and long

            if distance_km <= 1000:
                return "Short Haul"
            return "Long Haul"

        self.plan_df["haul_type"] = self.plan_df.apply(
            lambda row: classify_flight_distance(row["origin"], row["dest"]), axis=1
        )

        # categorise by quality of airlines
        callsigns = set(self.plan_df["callsign"])
        commercial = set()
        non_commercial = set()

        # ICAO callsign have 3 indicating the airlines and the next one must be a digit (for commercial ones)
        def is_valid_callsign(callsign):
            return callsign[:3].isalpha() and callsign[3].isdigit()

        for i in callsigns:
            if is_valid_callsign(i):
                commercial.add(i)
            else:
                non_commercial.add(i)

        self.plan_df = self.plan_df[
            (self.plan_df["milcivil"] == "C") | (self.plan_df["callsign"].apply(is_valid_callsign))
        ]  # as we don't care about military aircrafts

        def classify_class(row):
            if row["callsign"][:3] in commercial:
                return "Commercial"
            else:
                return "Non-Commercial"

        # categorise by "cheap", "mid" or expensive if you somehow get average fair costs per airlines, until then nope

        # Applying the title of column
        self.plan_df["Class"] = self.plan_df.apply(classify_class, axis=1)

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
