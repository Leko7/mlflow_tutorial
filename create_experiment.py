import mlflow
from mlflow_utils import create_mlflow_experiment

if __name__ == "__main__":

    experiment_id = create_mlflow_experiment(experiment_name="ocp7", artifact_location="log_reg_artifacts", tags={"env":"dev", "version": "1.0.0"})
    print(f"Experiment ID: {experiment_id}")