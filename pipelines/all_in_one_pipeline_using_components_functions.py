import sys
from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Model, Output

# Define custom base image built from project Dockerfile
BASE_IMAGE = "quay.io/xxxxx/training-image:latest"

# -------------------------
# STEP 1: Prepare Data Component
# -------------------------
@dsl.component(base_image=BASE_IMAGE)
def prepare_data(
    training_dataset: Output[Dataset], test_dataset: Output[Dataset]
):
    import sys
    sys.path.append("/app")  # Allows python to discover the /scripts package
    from scripts.all_in_one_ML_app import prepare_data as run_preparation
    # Directly execute the Python function
    run_preparation(
        training_output_path=training_dataset.path,
        test_output_path=test_dataset.path,
    )

# -------------------------
# STEP 2: Train Model Component
# -------------------------
@dsl.component(base_image=BASE_IMAGE)
def train_model(
    training_dataset: Input[Dataset], trained_model: Output[Model]
):
    import sys
    sys.path.append("/app")
    from scripts.all_in_one_ML_app import train_model as run_training
    # Directly execute the Python function
    run_training(
        training_dataset_path=training_dataset.path,
        model_path=trained_model.path,
    )

# -------------------------
# STEP 3: Evaluate Model Component
# -------------------------
@dsl.component(base_image=BASE_IMAGE)
def evaluate_model(
    test_dataset: Input[Dataset],trained_model: Input[Model],accuracy: dsl.OutputPath[float]
) -> float:
    import sys
    sys.path.append("/app")
    from scripts.all_in_one_ML_app import evaluate_model as run_evaluation
    # Directly execute the Python function and pass the accuracy output path
    accuracy = run_evaluation(
        test_dataset_path=test_dataset.path,
        model_path=trained_model.path,
        accuracy_path=accuracy
    )
    return accuracy

# -------------------------
# PIPELINE DEFINITION
# -------------------------
@dsl.pipeline(
    name="pipeline-functional-components",
    description="End-to-end pipeline using functional python imports",
    outputs={"model_accuracy": dsl.PipelineOutput(type=float)},
)
def ml_pipeline():
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

    # Return the accuracy
    return dsl.PipelineOutputs(
        model_accuracy=evaluation_phase.output
    )

# -------------------------
# PIPELINE COMPILATION
# -------------------------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline, package_path="ml_pipeline.yaml"
    )
    print(
        "Pipeline compilation successful! Output saved as 'ml_pipeline.yaml'"
    )
