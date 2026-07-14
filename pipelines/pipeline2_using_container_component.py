from kfp import dsl
from kfp import compiler

# Define custom base image built from project Dockerfile 
BASE_IMAGE = "quay.io/xxxxx/training-image:latest"

# -------------------------
# STEP 1: Prepare Data Component
# -------------------------
@dsl.container_component
def prepare_data(
    training_dataset: dsl.Output[dsl.Dataset], test_dataset: dsl.Output[dsl.Dataset]
):
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["python", "/app/scripts/prepare_data.py"],
        args=[training_dataset.path, test_dataset.path]
    )

# -------------------------
# STEP 2: Train Model Component
# -------------------------
@dsl.container_component
def train_model(
    training_dataset: dsl.Input[dsl.Dataset], trained_model: dsl.Output[dsl.Model]
):
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["python", "/app/scripts/train_model.py"],
        args=[training_dataset.path, trained_model.path]
    )

# -------------------------
# STEP 3: Evaluate Model Component
# -------------------------
@dsl.container_component
def evaluate_model(
    test_dataset: dsl.Input[dsl.Dataset], trained_model: dsl.Input[dsl.Model], accuracy: dsl.OutputPath[float]
):
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["python", "/app/scripts/evaluate_model.py"],
        args=[test_dataset.path, trained_model.path, accuracy]
    )

# -------------------------
# PIPELINE DEFINITION
# -------------------------
@dsl.pipeline(
    name="pipeline-container-components",
    description="End-to-end pipeline to prepare data, train model and evaluate accuracy", 
    outputs={"model_accuracy": dsl.PipelineOutput(type=str)} 
)
def ml_pipeline():
    # Prepare the data
    preparation_phase = prepare_data()
    
    # train the model
    training_phase = train_model(
        training_dataset=preparation_phase.outputs["training_dataset"]
    )
    
    # evaluate teh model
    evaluation_phase = evaluate_model(
        test_dataset=preparation_phase.outputs["test_dataset"],
        trained_model=training_phase.outputs["trained_model"]
    )

    # Return the accuracy
    return dsl.PipelineOutputs(
        model_accuracy=evaluation_phase.output
    )
# -------------------------
# PIPELINE COMPILATION
# -------------------------
if __name__ == "__main__":
    # Compile the pipeline definition into a YAML file
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline,
        package_path="ml_pipeline.yaml"
    )
    print("Pipeline compilation successful! Output saved as 'ml_pipeline.yaml'")
    