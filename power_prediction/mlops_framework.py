import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from power_prediction.util import get_project_root


def save_model(model_name, model, X_test, y_test, y_pred, param_dict, model_type="sklearn"):
    """
    Saves a model, its parameters, evaluation metrics to MLflow, and registers it to the Model Registry.

    Parameters:
        model_name (str): The name of the model to log and register.
        model: The trained model object.
        X_test: The test features for evaluating the model.
        y_test: The true values for the test set.
        y_pred: The predicted values for the test set.
        param_dict (dict): Dictionary containing parameter names and their values.
                           Format should be {param_name: value, ...}
        model_type (str): Type of model, e.g., "sklearn", "xgboost", "lightgbm", "keras", "pytorch".
                          Default is "sklearn".
    """

    # Set tracking URI to the local directory
    project_root = get_project_root()
    mlflow.set_tracking_uri(f"file:{project_root}/Models/mlruns")

    # Set signature
    signature = infer_signature(X_test, y_test)


    # Start a new run in MLflow
    with mlflow.start_run():
        # Log model name as a tag
        mlflow.set_tag("model_name", model_name)

        # Log model parameters
        for param_name, param_value in param_dict.items():
            mlflow.log_param(param_name, param_value)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAPE", mape)

        # Log the model based on the specified model type
        if model_type == "sklearn":
            model_info = mlflow.sklearn.log_model(model, "model", signature=signature)
        elif model_type == "xgboost":
            model_info = mlflow.xgboost.log_model(model, "model", signature=signature)
        elif model_type == "lightgbm":
            model_info = mlflow.lightgbm.log_model(model, "model", signature=signature)
        elif model_type == "keras":
            model_info = mlflow.keras.log_model(model, "model", signature=signature)
        elif model_type == "pytorch":
            model_info = mlflow.pytorch.log_model(model, "model", signature=signature)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")

        # Register model in the Model Registry
        if model_info:
            try:
                # Try registering the model (create new version if already exists)
                mlflow.register_model(model_info.model_uri, model_name)
            except Exception as e:
                print(f"Model registration failed: {e}")


def get_model(model_name, model_version):
    """Returns the trained model as saved"""
    project_root = get_project_root()
    mlflow.set_tracking_uri(f"file:{project_root}/Models/mlruns")
    model_uri = f"models:/{model_name}/{model_version}"
    return mlflow.pyfunc.load_model(model_uri)


def print_all_models():
    """Prints all registered models in the MLflow Model Registry along with their versions, parameters, and metrics."""

    project_root = get_project_root()
    mlflow.set_tracking_uri(f"file:{project_root}/Models/mlruns")
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()

    for model in models:
        print(f"Model Name: {model.name}")
        model_versions = client.search_model_versions(f"name='{model.name}'")

        for version in model_versions:
            print(f"  Version: {version.version}")
            run_info = client.get_run(version.run_id)
            print(f"    Run ID: {version.run_id}")

            print("    Parameters:")
            for param_name, param_value in run_info.data.params.items():
                print(f"      {param_name}: {param_value}")

            print("    Metrics:")
            for metric_name, metric_value in run_info.data.metrics.items():
                print(f"      {metric_name}: {metric_value}")

            print("-" * 50)
