import mlflow


MODEL_NAME = "CancerRiskModel"


def main():

    model_uri = (
        f"models:/{MODEL_NAME}@production"
    )

    print(
        f"Loading model from: {model_uri}"
    )

    model = mlflow.pyfunc.load_model(
        model_uri
    )

    print(
        "\nProduction model loaded successfully."
    )

    print(
        f"Model type: {type(model)}"
    )


if __name__ == "__main__":
    main()