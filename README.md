# Experiment Library and Setup (ELIAS)

## 1. Main functionalities

### `Config` class
Intuitive dataclass extension that addresses issues commonly encountered in research projects. 
A `Config` has the following features:
 - Easily **persistable** to a file (JSON or YAML)
 - Stored configs are **human-readable and editable**
 - Special support for storing **enums** and **inheritance structures**
 - **Backward compatibility** to allow loading older persisted configs

TODO: Overview image of benefits of using Config over regular dataclass

Philosophy for stored config file types:
 - Everything should be human-readable
 - JSON for configs that may be viewed often (statistics, preprocessing/training configs, evaluation results)
 - YAML for configs that have to be edited (run specifications)

### Experiment Workflow Utilities
To speed up the experimentation process with utility classes the `elias` library assumes the following workflow:
 
Stage | Input | Output | Utility
---|---|---|---
Data Preprocessing | <ul><li>Processing Configuration</li> <li>Raw Data</li></ul> | <ul><li>Preprocessed Data (.p, .p.gz, .json, .npy, ...)</li> <li>Data Statistics (stats.json)</li> <li>Preprocessing Config (config.json)</li></ul> | DataFolder -> DataManager
Training/Fitting | <ul><li>Preprocessed Data</li> <li>Hyperparameters</li></ul> | <ul><li>Model checkpoints</li> <li>Hyperparameter configs</li></ul> | ModelManager -> RunManager
Evaluation | <ul><li>Trained model</li><li>Evaluation Config</li></ul> | <ul><li>Evaluation Config</li> <li>Evaluation Results</li></ul> | RunManager -> EvaluationManager
Manual Analysis | Any model/data | Plots, statistics, images | AnalysisFolder -> AnalysisManager


