import json

def prepare_tuning():
    metadata = {
        "id": "tuannm3812/stellar-classification-hyperparameter-tuning",
        "title": "Stellar Classification - Hyperparameter Tuning",
        "code_file": "04_hyperparameter_tuning.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [
            "playground-series-s6e6"
        ],
        "kernel_sources": []
    }
    with open("notebooks/kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Updated kernel-metadata.json to hyperparameter tuning configuration.")

if __name__ == "__main__":
    prepare_tuning()
