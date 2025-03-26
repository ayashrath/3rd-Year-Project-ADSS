from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

from toolkit_dn import Dataset


def plot_predictions_vs_actual(self, y_pred, n_samples=1000, title="Predictions vs Actual"):
# Get the first n_samples in original order
    indices = range(len(y_pred))
    y_true_subset = self.y_test.ravel()
    y_pred_subset = y_pred
    
    plt.figure(figsize=(15, 6))
    
    # Plot true values as points
    plt.scatter(
        indices, 
        y_true_subset, 
        color='blue', 
        alpha=0.5,
        s=10,  # Smaller point size
        label='Actual Values'
    )
    
    # Plot predicted values as line
    plt.plot(
        indices, 
        y_pred_subset, 
        color='red', 
        linewidth=1.5,
        label='Predicted Values'
    )
    
    plt.xlabel('Sample Index (original order)')
    plt.ylabel('Turnaround Time (scaled)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Add the method to your Dataset class
Dataset.plot_predictions_vs_actual = plot_predictions_vs_actual


def random_forest(self, n_estimators=100, max_depth=None, random_state=314, verbose=1):
    """
    Train and evaluate a Random Forest regressor on the dataset.
    
    Parameters:
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum depth of the trees
    - random_state: Random seed for reproducibility
    - verbose: Controls verbosity of output
    
    Returns:
    - Dictionary containing trained model and evaluation metrics
    """
    if not self.clean_bool:
        raise ValueError("Dataset must be cleaned before training. Call clean() first.")
    
    # Initialize Random Forest model
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        verbose=verbose,
        n_jobs=-1  # Use all available cores
    )
    
    # Train the model
    if verbose:
        print("Training Random Forest model...")
    rf.fit(self.x_train, self.y_train.ravel())
    
    # Make predictions
    y_train_pred = rf.predict(self.x_train)
    y_test_pred = rf.predict(self.x_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'mae': mean_absolute_error(self.y_train, y_train_pred),
            'mse': mean_squared_error(self.y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'r2': r2_score(self.y_train, y_train_pred)
        },
        'test': {
            'mae': mean_absolute_error(self.y_test, y_test_pred),
            'mse': mean_squared_error(self.y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'r2': r2_score(self.y_test, y_test_pred)
        }
    }
    
    if verbose:
        print("\nTraining Metrics:")
        print(f"MAE: {metrics['train']['mae']:.4f}")
        print(f"MSE: {metrics['train']['mse']:.4f}")
        print(f"RMSE: {metrics['train']['rmse']:.4f}")
        print(f"R²: {metrics['train']['r2']:.4f}")
        
        print("\nTest Metrics:")
        print(f"MAE: {metrics['test']['mae']:.4f}")
        print(f"MSE: {metrics['test']['mse']:.4f}")
        print(f"RMSE: {metrics['test']['rmse']:.4f}")
        print(f"R²: {metrics['test']['r2']:.4f}")
    
    return {
        'model': rf,
        'metrics': metrics,
        'feature_importances': rf.feature_importances_
    }

# Add the method to your Dataset class
Dataset.random_forest = random_forest
# Load and clean your dataset
ds = Dataset()
ds.clean()

# Train Random Forest
rf_results = ds.random_forest(n_estimators=500, max_depth=100, verbose=1)

y_test_pred = rf_results['model'].predict(ds.x_test)

# Plot the comparison
ds.plot_predictions_vs_actual(
    y_test_pred, 
    n_samples=500, 
    title="Random Forest Predictions vs Actual Values"
)