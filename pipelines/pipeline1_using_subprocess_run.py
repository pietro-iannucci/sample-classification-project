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
    import subprocess
    import sys
    # Execute the python script inside the container 
    result = subprocess.run(
        [
            "python",
            "/app/scripts/prepare_data.py",
            training_dataset.path,
            test_dataset.path,
        ],
        capture_output=True,
        text=True,
        check=True
    )
    # Standard out forwarding to make container logs visible in OpenShift AI UI
    print(result.stdout)

# -------------------------
# STEP 2: Train Model Component
# -------------------------
@dsl.component(base_image=BASE_IMAGE)
def train_model(
    training_dataset: Input[Dataset], trained_model: Output[Model]
):
    import subprocess
    import sys
    # Execute the python script inside the container 
    result = subprocess.run(
        [
            "python",
            "/app/scripts/train_model.py",
            training_dataset.path,
            trained_model.path,
        ],
        capture_output=True,
        text=True,
        check=True
    )
    # Standard out forwarding to make container logs visible in OpenShift AI UI
    print(result.stdout)

# -------------------------
# STEP 3: Evaluate Model Component
# -------------------------
@dsl.component(base_image=BASE_IMAGE)
def evaluate_model(
    test_dataset: Input[Dataset], trained_model: Input[Model], accuracy: dsl.OutputPath[float]
) :
    import subprocess
    import sys
    # Execute the python script inside the container 
    result = subprocess.run(
        [
            "python",
            "/app/scripts/evaluate_model.py",
            test_dataset.path,
            trained_model.path,
            accuracy         
        ],
        capture_output=True,
        text=True,
        check=True
    )
    # Standard out forwarding to make container logs visible in OpenShift AI UI
    print(result.stdout)
       
# -------------------------
# PIPELINE DEFINITION
# -------------------------
@dsl.pipeline(
    name="pipeline-subprocess-components",
    description="End-to-end pipeline to prepare data, train model and evaluate accuracy using subprocess",
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
        trained_model=training_phase.outputs["trained_model"]
    )

    # Return the accuracy
    return dsl.PipelineOutputs(
        model_accuracy=evaluation_phase.outputs["accuracy"]
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
