import json

with open("notebooks/03_model_tuning_and_ensemble.ipynb", "r") as f:
    data = json.load(f)

new_source = [
    "# Dynamic path resolution for Local vs Kaggle environment\n",
    "if os.path.exists('/kaggle/input/competitions/playground-series-s6e6'):\n",
    "    DATA_DIR = '/kaggle/input/competitions/playground-series-s6e6'\n",
    "    PRED_DIR = '/kaggle/input/notebooks/tuannm3812/stellar-classification-baseline-modeling/predictions'\n",
    "    OUTPUT_DIR = '.'\n",
    "else:\n",
    "    DATA_DIR = '../data'\n",
    "    PRED_DIR = '../predictions'\n",
    "    OUTPUT_DIR = '..'\n",
    "\n",
    "print(f\"Data Directory: {DATA_DIR}\")\n",
    "print(f\"Predictions Directory: {PRED_DIR}\")\n",
    "print(f\"Output Directory: {OUTPUT_DIR}\")"
]

data['cells'][3]['source'] = new_source

with open("notebooks/03_model_tuning_and_ensemble.ipynb", "w") as f:
    json.dump(data, f, indent=1)

print("Updated config successfully!")
