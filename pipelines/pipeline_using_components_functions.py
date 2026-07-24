from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Model, Output

# Define custom base image built from project Dockerfile
BASE_IMAGE = "quay.io/xxxxx/training-image:latest" 

# -------------------------------------------- 
# STEP 1: Define Prepare Data Component 
# --------------------------------------------
@dsl.component(base_image=BASE_IMAGE)
def prepare_data(training_dataset: Output[Dataset], test_dataset: Output[Dataset]):
    from scripts.prepare_data import prepare_data as run_preparation
    run_preparation(
        training_output_path=training_dataset.path,
        test_output_path=test_dataset.path,
    )

# -------------------------------------------- 
# STEP 2: Define Train Model Component 
# --------------------------------------------
@dsl.component(base_image=BASE_IMAGE)
def train_model(training_dataset: Input[Dataset], trained_model: Output[Model]):
    from scripts.train_model  import train_model as run_training
    run_training(
        training_dataset_path=training_dataset.path,
        model_path=trained_model.path,
    )

# -------------------------------------------- 
# STEP 3: Define Evaluate Model Component 
# --------------------------------------------
@dsl.component(base_image=BASE_IMAGE)
def evaluate_model(test_dataset: Input[Dataset],trained_model: Input[Model],accuracy: dsl.OutputPath[float]) -> float:
    from scripts.evaluate_model import evaluate_model as run_evaluation
    accuracy = run_evaluation(
        test_dataset_path=test_dataset.path,
        model_path=trained_model.path,
        accuracy_path=accuracy
    )
    return accuracy

# -------------------------------------------- 
# STEP 4: Define Save Model Component 
# --------------------------------------------
@dsl.component(base_image=BASE_IMAGE)
def save_model(trained_model: Input[Model], model_registry_path: str):
    from scripts.save_model import save_model as run_save_model
    run_save_model(
        model_path=trained_model.path,
        model_registry_path=model_registry_path,
    )

# -------------------------------------------------------- 
# STEP 5: define a component that sends metrics to MLFlow
# --------------------------------------------------------
@dsl.component(base_image=BASE_IMAGE)
def log_metrics(accuracy: float, tracking_uri: str,experiment_name: str):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)

# -------------------------------------------- 
# PIPELINE DEFINITION 
# --------------------------------------------
@dsl.pipeline(
    name="pipeline-functional-components",
    description="End-to-end pipeline using functional python imports",
    outputs={"model_accuracy": dsl.PipelineOutput(type=float)},
)
def ml_pipeline(
    environment: str = "development",
    model_registry_path: str = "/model-registry",
):
    # Prepare the data
    preparation_phase = prepare_data()
    
    # Train the model
    training_phase = train_model(
        training_dataset=preparation_phase.outputs["training_dataset"]
    )
    
    # Evaluate the model
    evaluation_phase = evaluate_model(
        test_dataset=preparation_phase.outputs["test_dataset"],
        trained_model=training_phase.outputs["trained_model"],
    )
    
    # Send accuracy metric to MLFlow
    log_metrics(
        accuracy=evaluation_phase.output,
        tracking_uri=mlflow_tracking_uri,
        experiment_name="Fraud Detection",
    )
    
    # Save  model in registry only if training succeeds and the pipeline runs in training environment
    with dsl.If(
        (evaluation_phase.output > 0.8) &
        (environment == "training")
    ):
        save_model(
            trained_model=training_phase.outputs["trained_model"],
            model_registry_path=model_registry_path,
        )
    
    # Return the accuracy
    return dsl.PipelineOutputs(
        model_accuracy=evaluation_phase.output
    )

# -------------------------------------------- 
# PIPELINE COMPILATION 
# --------------------------------------------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline, package_path="ml_pipeline.yaml"
    )
    print(
        "Pipeline compilation successful! Output saved as 'ml_pipeline.yaml'"
    )
