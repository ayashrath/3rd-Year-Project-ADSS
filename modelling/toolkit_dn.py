"""
Makes life easier by:
- having stuff for importing datasets,
- train model
- producing Test Output Against Test
...
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm, notebook
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.utils.data.dataloader as tudd  # Just for type casting
from IPython.display import display  # incase it is jupyter notebook
from typing import List, Tuple
from datetime import datetime

"""
LabelEncoder - Converts Categorical Data into Numerical Ones
MinMaxScaler = (x - min_x) / (max_x - min_x)
MaxAbsScaler = x / max_abs_x
RobustScaler = (x - median_x) / IQR_x {IQR_x = Q3_x - Q1_x (Q1_x and Q3_x are the 1st and 3rd quartiles of x}
StandardScalar = (x - mean_x) / (std_dev)
"""


class Dataset:
    """
    Basically a Dataset Handelling Class

    Btw just note - if there is something you want to do with the df but can't as there is not an
    accessor for it, just access the object var :|
    """
    def __init__(
        self, train: str = "train.csv", test: str = "test.csv",
        loading_status: str = True, remove_threshold: float = 0.35
    ):
        self.train_df = pd.read_csv(train)
        self.test_df = pd.read_csv(test)
        self.clear_status = False
        self.remove_threshold = remove_threshold  # removes all cols which have less than this % of total data
        self.clean_bool = False
        pd.set_option('display.max_columns', None)
        if loading_status:
            print("Dataset Loaded!")

    def __str__(self):
        print(self.train_df)
        print(self.test_df)

    def fetch_tt_nth_percentile(self, n: int):
        if self.clean_bool:
            raise RuntimeError("Run before clean")
        temp_df = pd.concat([self.train_df, self.test_df], ignore_index=True)
        tt = np.array(temp_df["turnaroundtime_s"])
        return np.percentile(tt, n)

    def update_train_test(self, num, operator: str):
        """
        < means all enties less than
        > means all entries greater than
        """
        if self.clean_bool:
            val = "tt_s"
        else:
            val = "turnaroundtime_s"

        if operator == "<":
            self.train_df = self.train_df[self.train_df[val] < num]
            self.test_df = self.test_df[self.test_df[val] < num]
        elif operator == ">":
            self.train_df = self.train_df[self.train_df[val] > num]
            self.test_df = self.test_df[self.test_df[val] > num]
        else:
            raise AttributeError("It only supports < or >")

    def fetch_info(self):
        return self.train_df.info()

    def fetch_train(self):
        return self.train_df

    def current_clean_status(self):
        return self.clear_status

    def fetch_test(self):
        return self.test_df

    def fetch_cols_train(self, cond):
        return self.train_df[cond]

    def filter_cols_train(self, num, operator: str):
        """
        <= means all enties less than
        > means all entries greater than
        """
        if self.clean_bool:
            val = "tt_s"
        else:
            val = "turnaroundtime_s"

        if operator == "<":
            return self.train_df[self.train_df[val] <= num]
        elif operator == ">":
            return self.train_df[self.train_df[val] > num]
        else:
            raise AttributeError("It only supports < or >")

    def write_train_file(self, path: str):
        self.train_df.to_csv(path)

    def isin_cols_train(self, vals):
        return self.train_df[self.train_df.isin(vals)]

    def clean(self, remove_outlier_tt=True, remove_upper_lower_thresh: int = 1):
        # consts
        drop_cols = ["Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0", "datetime", "start_time"]
        rename_cols = {
            "flight_type": "international",
            "haul_type": "long_haul",
            "mode_s_hex": "hex",
            "turnaroundtime_s": "tt_s",
            "cloud_height_layer_0": "cloud_h_lay_0",
            "cloud_quant_layer_0": "cloud_det_lay_0",
            "Class": "type"
        }
        le_dict = {
            "international": LabelEncoder(),
            "long_haul": LabelEncoder(),
            "hex": LabelEncoder(),
            "callsign": LabelEncoder(),
            "cloud_det_lay_0": LabelEncoder(),
            "type": LabelEncoder(),
        }

        for df in [self.train_df, self.test_df]:
            # drop stuff
            df.drop(columns=drop_cols, inplace=True)

            # binary -> num
            df.rename(columns=rename_cols, inplace=True)

            # encode all the non int/float stuff :)
            for col in ["hex", "callsign", "international", "long_haul", "cloud_det_lay_0", "type"]:
                if df is self.train_df:
                    # so there would exist a case handling for unknown flights
                    df[col] = le_dict[col].fit_transform(df[col].fillna("Unknown"))
                else:
                    # so that there same mapping for train and test - creates dict - label:id_assigned
                    label_map = {label: idx for idx, label in enumerate(le_dict[col].classes_)}
                    # type int as map can make it float series to allow NaN
                    df[col] = df[col].fillna("Unknown").map(label_map).fillna(-1).astype(int)

            # break time (dt.stuff -> date time manager in pd)
            df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")  # NaT for not time returned
            df["end_datetime"] = pd.to_datetime(df["end_datetime"], errors="coerce")

            for prefix in ["start", "end"]:
                df[f"{prefix} year"] = df[f"{prefix}_datetime"].dt.year
                df[f"{prefix} month"] = df[f"{prefix}_datetime"].dt.month
                df[f"{prefix} day"] = df[f"{prefix}_datetime"].dt.day
                df[f"{prefix} hour"] = df[f"{prefix}_datetime"].dt.hour
                df[f"{prefix} minute"] = df[f"{prefix}_datetime"].dt.minute
                df[f"{prefix} second"] = df[f"{prefix}_datetime"].dt.second

            df.drop(columns=["start_datetime", "end_datetime"], inplace=True)

            # remove field with < remove_threshold% of the dataset length
            df.dropna(axis=1, thresh=int(self.remove_threshold * len(df)), inplace=True)

            # fill empty with 0
            df.fillna(0, inplace=True)

        outlier_thresh_upper = self.fetch_tt_nth_percentile(100-remove_upper_lower_thresh)
        self.update_train_test(outlier_thresh_upper, "<")
        outlier_thresh_lower = self.fetch_tt_nth_percentile(remove_upper_lower_thresh)
        self.update_train_test(outlier_thresh_lower, ">")

        self.clean_bool = True

        return le_dict  # in the case there is a need for decoding

    def drop_cols(self, cols_lst: List[str]):
        self.train_df.drop(columns=cols_lst)
        self.test_df.drop(columns=cols_lst)

    def format_data(self, scalar_type: str = "minmax", save_scalar_val: bool = True, batch_size: int = 32) -> Tuple:

        # X = input, Y = output
        x_train = self.train_df.drop(columns=["tt_s"]).values  # convert into np mat
        y_train = self.train_df["tt_s"].values.reshape(-1, 1)  # to make it a mat, else will be just an array
        x_test = self.test_df.drop(columns=["tt_s"]).values
        y_test = self.test_df["tt_s"].values.reshape(-1, 1)

        # The label and features should not share the same scalar to maininatin thier distributions
        # All features should use the same scalar to mainintain their relationship
        # The test and train should have same scalar to ensure consistency - else model will fuck up (mind my French!)
        if "minmax" in scalar_type.lower() and "abs" not in scalar_type.lower():
            scalar_feature = MinMaxScaler()
            scalar_label = MinMaxScaler()
        elif "maxabs" in scalar_type.lower():
            scalar_feature = MaxAbsScaler()
            scalar_label = MaxAbsScaler()
        elif "robust" in scalar_type.lower():
            scalar_feature = RobustScaler()
            scalar_label = RobustScaler()
        elif "standard" in scalar_type.lower():
            scalar_feature = StandardScaler()
            scalar_label = StandardScaler()
        else:
            raise AttributeError("This kind of scalar is not supported, idiot :)")

        x_train = scalar_feature.fit_transform(x_train)
        y_train = scalar_label.fit_transform(y_train)
        x_test = scalar_feature.fit_transform(x_test)
        y_test = scalar_label.fit_transform(y_test)

        # saving it as important whenever you might want to deploy it
        if save_scalar_val:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            joblib.dump(scalar_feature, f'./stores/feature_scaler_{timestamp}.joblib')
            joblib.dump(scalar_label, f'./stores/label_scaler_{timestamp}.joblib')

        # create the tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        except RuntimeError as e:
            print(f"CUDA error: {e}")
            x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Use TensorDatasets to make datasets and then Create Dataloaders (does batch and shuffling as needed)
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)  # order can exist

        # for inp dimension, this is the simplest way to do it here
        return (train_loader, test_loader, x_train_tensor.shape[1], scalar_label)


class ModelTrainerDNN:
    """
    Just trains the model for you
    """
    def __init__(
        self, model: nn.Module, train_loader: tudd.DataLoader, test_loader: tudd.DataLoader, label_scalar, criteria,
        optimiser, epoch: int, add_training_patience: bool = True, patience_thresh_epoch_percent: float = 0.1,
        jupter_notebook: bool = False, auto_save_model: bool = True
    ):
        """
        Note - Use consts defined here instead of remembering level of valid
        And if its jupyter notebook then stuff like graphs, stuff like validation values, scalar values will not be
        saved and will be just printed as that is just more logical.
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criteria = criteria  # annoying to typecast so didn't - so rely on proper errors :)
        self.optimiser = optimiser  # same here
        self.epoch = epoch
        self.add_patience = add_training_patience  # way to stop training if the model has converged well enough
        self.patience_thresh_percent = patience_thresh_epoch_percent  # epoch * thresh = patience
        self.notebook = jupter_notebook
        self.label_scalar = label_scalar
        self.auto_save_model = auto_save_model

    def update_epoch(self, epoch):
        self.epoch = epoch

    def update_patience_thresh(self, thresh):
        self.patience_thresh_percent = thresh

    def update_criteria(self, criteria):
        self.criteria = criteria

    def update_optimiser(self, optimiser):
        self.optimiser = optimiser

    def save_onnx(self):
        """
        My suggestion - use netron to view the file
        """
        # incase you run it without training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # gets feature count
        first_sample, _ = next(iter(self.train_loader))
        feature_count = first_sample.shape[1]

        # gets batch size
        batch_size = self.train_loader.batch_size

        # a dummy data to pass though for onnx
        dummy_input = torch.randn(batch_size, feature_count).to(device)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                f'./stores/dnn_{timestamp}.onnx',
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},  # for variable batch size
                    "output": {0: "batch_size"}  # Same
                }
            )
        except Exception:
            print("Error in exporting to ONNX!!")

    def save_model(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self.model.state_dict(), f'./stores/dnn_{timestamp}.pth')

    def train_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model.to(device)
        except RuntimeError as e:
            print(f"CUDA error: {e}")
            device = torch.device("cpu")
            self.model.to(device)

        # patience init
        if self.add_patience:
            min_loss = float("inf")  # start val loss
            patience = self.epoch * self.patience_thresh_percent
            patience_counter = 0

        print("\n")
        # create progress
        if self.notebook:
            epoch_range = notebook.tqdm(range(self.epoch))
        else:
            epoch_range = tqdm(range(self.epoch))

        loss_lst = []
        len_batch_size = len(self.train_loader)

        # loop
        for epoch in epoch_range:  # add loss into tqdm
            self.model.train()  # train mode
            total_loss = 0  # all batches loss

            for x_batch, y_batch in self.train_loader:
                try:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                except RuntimeError as e:
                    print(f"CUDA error: {e}")
                    x_batch, y_batch = x_batch, y_batch

                self.optimiser.zero_grad()
                predictions = self.model(x_batch)
                loss = self.criteria(predictions, y_batch)
                loss.backward()
                self.optimiser.step()
                total_loss += loss.item()  # add single value tensor loss for batch to total

            mean_loss = total_loss/len_batch_size
            loss_lst.append(mean_loss)
            epoch_range.set_postfix(loss=mean_loss)

            if self.add_patience:
                if total_loss < min_loss:
                    min_loss = total_loss
                    patience_counter = 0  # reset if lower
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        epoch_range.close()
                        print("Patience Exceeded!")
                        break

        if self.auto_save_model:
            self.save_model()

        return loss_lst

    def validate(self, loss_lst, simple: bool = True, train: bool = False, display_df: bool = False):
        """
        Can be shortened :(
        """
        if not train:
            print("\nTesting Data Validation")
            # validate
            self.model.eval()

            test_loss = 0
            y_true_test = []
            y_predicted_test = []

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with torch.no_grad():
                for x_batch, y_batch in self.test_loader:
                    try:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    except RuntimeError as e:
                        print(f"CUDA error: {e}")
                        x_batch, y_batch = x_batch, y_batch

                    predictions = self.model(x_batch)
                    test_loss += self.criteria(predictions, y_batch).item()

                    # collect actual & predicted values
                    y_true_test.extend(y_batch.cpu().numpy())
                    y_predicted_test.extend(predictions.cpu().numpy())

            # Processing
            y_true_test = np.array(y_true_test).flatten()
            y_predicted_test = np.array(y_predicted_test).flatten()

            # Descale
            y_true_test = self.label_scalar.inverse_transform(y_true_test.reshape(-1, 1)).flatten()
            y_predicted_test = self.label_scalar.inverse_transform(y_predicted_test.reshape(-1, 1)).flatten()

            # Scores
            mae = mean_absolute_error(y_true_test, y_predicted_test)
            mse = mean_squared_error(y_true_test, y_predicted_test)
            r2 = r2_score(y_true_test, y_predicted_test)

            print("==============================")
            print("Stat Metrics")
            print("==============================")
            print(f"MAE Value: {mae}")
            print(f"MSE Value: {mse}")
            print(f"R² Value: {r2}")

            if not simple:
                if display_df:
                    # Summary df
                    print("==============================")
                    print("Prediction DF")
                    print("==============================")

                    error_abs = np.abs(y_true_test - y_predicted_test)
                    summary_df = pd.DataFrame({
                        'True Value': y_true_test,
                        'Predicted Value': y_predicted_test,
                        'Absolute Error': error_abs
                    })

                    if self.notebook:
                        display(summary_df)
                    else:
                        print(summary_df)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Loss Progression - Training
                plt.figure(figsize=(30, 10))
                plt.plot(range(1, len(loss_lst) + 1), loss_lst, label="Training Loss")
                plt.xlabel("Index")
                plt.ylabel("Loss")
                plt.title("Training Loss Progression")

                if self.notebook:
                    plt.show()
                else:
                    plt.savefig(f"./stores/training_loss_graph_{timestamp}.jpg")

                # Accuracy Graph
                plt.figure(figsize=(30, 10))
                plt.scatter(y_true_test, y_predicted_test, label="true vs predicted")  # ascending order and y = x
                plt.plot([y_true_test.min(), y_true_test.max()], [y_true_test.min(), y_true_test.max()], "--r", "1")
                plt.xlabel("True Value")
                plt.ylabel("Predicted Value")
                plt.title("Accuracy Graph (y=x is 100%)")
                plt.legend()

                if self.notebook:
                    plt.show()
                else:
                    plt.savefig(f"./stores/test_accuracy_graph_{timestamp}.jpg")

                # TT Behaviour Graph
                plt.figure(figsize=(30, 10))
                plt.scatter(np.arange(len(y_true_test)), y_true_test, label="True Value")  # ascending order
                plt.plot(np.arange(len(y_true_test)), y_predicted_test, "r", label="Predicted Value")
                plt.xlabel("Index")
                plt.ylabel("Time (min)")
                plt.title("Curve of TT and Predicted TT functions")
                plt.legend()

                if self.notebook:
                    plt.show()
                else:
                    plt.savefig(f"./stores/test_tt_curve_graph_{timestamp}.jpg")

        else:
            print("\nTraining Data Validation")
            # validate
            self.model.eval()

            train_loss = 0
            y_true_train = []
            y_predicted_train = []

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with torch.no_grad():
                for x_batch, y_batch in self.train_loader:
                    try:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    except RuntimeError as e:
                        print(f"CUDA error: {e}")
                        x_batch, y_batch = x_batch, y_batch

                    predictions = self.model(x_batch)
                    train_loss += self.criteria(predictions, y_batch).item()

                    # collect actual & predicted values
                    y_true_train.extend(y_batch.cpu().numpy())
                    y_predicted_train.extend(predictions.cpu().numpy())

            # Processing
            y_true_train = np.array(y_true_train).flatten()
            y_predicted_train = np.array(y_predicted_train).flatten()

            # Descale
            y_true_train = self.label_scalar.inverse_transform(y_true_train.reshape(-1, 1)).flatten()
            y_predicted_train = self.label_scalar.inverse_transform(y_predicted_train.reshape(-1, 1)).flatten()

            # Scores
            mae = mean_absolute_error(y_true_train, y_predicted_train)
            mse = mean_squared_error(y_true_train, y_predicted_train)
            r2 = r2_score(y_true_train, y_predicted_train)

            print("==============================")
            print("Stat Metrics")
            print("==============================")
            print(f"MAE Value: {mae}")
            print(f"MSE Value: {mse}")
            print(f"R² Value: {r2}")

            if not simple:
                if display_df:
                    # Summary df
                    print("==============================")
                    print("Prediction DF")
                    print("==============================")

                    error_abs = np.abs(y_true_train - y_predicted_train)
                    summary_df = pd.DataFrame({
                        'True Value': y_true_train,
                        'Predicted Value': y_predicted_train,
                        'Absolute Error': error_abs
                    })

                    if self.notebook:
                        display(summary_df)
                    else:
                        print(summary_df)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Loss Progression - Training
                plt.figure(figsize=(30, 10))
                plt.plot(range(1, len(loss_lst) + 1), loss_lst, label="Training Loss")
                plt.xlabel("Index")
                plt.ylabel("Loss")
                plt.title("Training Loss Progression")

                if self.notebook:
                    plt.show()
                else:
                    plt.savefig(f"./stores/train_loss_graph_{timestamp}.jpg")

                # Accuracy Graph
                plt.figure(figsize=(30, 10))
                plt.scatter(y_true_train, y_predicted_train, label="true vs predicted")  # ascending order and y = x
                plt.plot([y_true_train.min(), y_true_train.max()], [y_true_train.min(), y_true_train.max()], "--r", "1")
                plt.xlabel("True Value")
                plt.ylabel("Predicted Value")
                plt.title("Accuracy Graph (y=x is 100%)")
                plt.legend()

                if self.notebook:
                    plt.show()
                else:
                    plt.savefig(f"./stores/train_accuracy_graph_{timestamp}.jpg")

                # TT Behaviour Graph
                plt.figure(figsize=(30, 10))
                plt.scatter(np.arange(len(y_true_train)), y_true_train, label="True Value")  # ascending order
                plt.plot(np.arange(len(y_true_train)), y_predicted_train, "r", label="Predicted Value")
                plt.xlabel("Index")
                plt.ylabel("Time (min)")
                plt.title("Curve of TT and Predicted TT functions")
                plt.legend()

                if self.notebook:
                    plt.show()
                else:
                    plt.savefig(f"./stores/train_tt_curve_graph_{timestamp}.jpg")
    
    def compute_feature_importance_pfi(self, n_shuffles: int = 10, criteria: str = "MSE", plot: bool = True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        x_test, y_test = [], []
        for x_batch, y_batch in self.test_loader:
            x_test.append(x_test)
            y_test.append(y_test)
        x_test, y_test = torch.cat(x_test), torch.cat(y_test)

        # normal score
        with torch.no_grad():
            init_score = self._compute_metric(self.model(x_test), y_test, criteria)

        # main stuff
        results = {}
        for feature_idx in range(x_test.shape[1]):
            shuffled_scores = []
            for _ in range(n_shuffles):
                x_shuffled = x_test.clone()
                x_shuffled[:, feature_idx] = x_shuffled[torch.randperm(x_shuffled.shape[0]), feature_idx]

                with torch.no_grad():
                    score = self._compute_metric(self.model(x_shuffled), y_test, criteria)
                shuffled_scores.append(init_score - score)

            results[f"feature_{feature_idx}"] = (np.mean(shuffled_scores), np.std(shuffled_scores))

        if plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sorted_results = dict(sorted(results.items(), key=lambda x: x[1][0], reverse=True))
            plt.figure(figsize=(8, min(6, len(results)//2)))
            plt.barh(
                list(sorted_results.keys()), 
                [x[0] for x in sorted_results.values()],
                xerr=[x[1] for x in sorted_results.values()],
                color='skyblue'
            )
            plt.title(f"PFI ({criteria.upper()}, Δ={init_score:.3f})")
            plt.xlabel("Importance (Score Decrease)")
            plt.tight_layout()
            plt.savefig(f"./stores/pfi_{timestamp}.png", bbox_inches='tight')
            plt.close()

        return results

    def _compute_metric(self, preds: torch.Tensor, target: torch.Tensor, metric: str):
        y_true, y_pred = target.cpu().numpy(), preds.cpu().numpy()
        if metric.lower() == "mse":
            return mean_squared_error(y_true, y_pred)
        elif metric.lower() == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif metric.lower() == "r2":
            return r2_score(y_true, y_pred)
        raise ValueError(f"Unknown metric: {metric}")

    def load_model(self, model_path: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path), map_location=device)
        self.model.to(device)


if __name__ == "__main__":
    datasets = Dataset()
    datasets.clean()
    print(datasets.fetch_info())
