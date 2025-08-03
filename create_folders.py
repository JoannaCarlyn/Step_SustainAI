import os

folders = [
    "data/raw", "data/processed", "data/synthetic",
    "model/energy_predictor", "model/nlp_transformer", "model/anomaly_detector", "model/prompt_optimizer",
    "src/gui", "src/nlp", "src/prediction", "src/optimization", "src/anomaly", "src/utils",
    "reports/visualizations",
    "documentation/readme_images",
    "notebooks/training_notebooks", "notebooks/experiments",
    "tests"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create base files
files = [
    "README.md", "requirements.txt", "run.sh", ".gitignore",
    "src/gui/app.py", "src/gui/layout.py",
    "src/nlp/parser.py", "src/nlp/complexity_score.py", "src/nlp/simplifier.py",
    "src/prediction/estimator.py", "src/optimization/recommender.py", "src/anomaly/detector.py",
    "src/utils/logger.py", "src/utils/config.py",
    "tests/test_nlp.py", "tests/test_predictor.py", "tests/test_gui.py"
]

for file in files:
    with open(file, "w"): pass  # create empty files
