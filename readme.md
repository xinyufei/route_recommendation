# Project README

## Overview
This project provides tools and algorithms for analyzing and 
visualizing data related to bandit problems, static modeling, 
and other computational experiments. 
It includes Python scripts for core functionality, 
Jupyter notebooks for testing, and visualizations to aid in 
interpreting results. 

*Please ignore code files corresponding to "game" because 
it is designed for previous other project's experiments.*

---

## File Structure

### `code/`
- **`bandit.py`**: Implements bandit algorithms for decision-making problems.
- **`static_model.py`**: Contains code for static modeling of given datasets.
- **`algorithm_for_game.py`**: Algorithms for game-based computations.
- **`read_data.py`**: Handles data reading and preprocessing.
- **`auxiliary.py`**: Provides utility functions to support the main scripts.

### `to_cpp/`
- **`wrapper.cpp`**: C++ wrapper for interfacing with Python.
- **`algorithm_for_game.py`**: Duplicate or adapted version for specific scenarios.
- **`ref.txt`**: Reference file for documentation or parameters.

### `test_notebook/`
- **`test_static_model.ipynb`**: Jupyter notebook for testing static modeling.
- **`test_bandit.ipynb`**: Notebook for testing bandit algorithms.
- **`test_algorithm_for_game.ipynb`**: Notebook for validating game-related algorithms.
- Log files (`*.log`) for recording test outputs.

### `static_figure/`
Contains static visualizations:
- **`sioux_falls_outsample_obj_zoomin.png`**: Zoomed-in view of out-sample objectives.
- Other figures related to `sioux_falls` data analysis.

### `bandit_figure/`
Includes visualizations related to bandit algorithm performance:
- **`sioux_falls_regret_trust_0p3_ucb_two_reward.png`**: Performance of UCB algorithms under specific trust parameters.
- Other trust and regret comparisons.

---

## Installation
1. Clone the repository or extract the ZIP file.
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running Core Scripts
Run any script in the `code/` directory by executing:
```bash
python code/<script_name>.py
```

### Testing with Jupyter Notebooks
1. Navigate to the `test_notebook/` folder.
2. Open the desired notebook using Jupyter:
   ```bash
   jupyter notebook test_notebook/<notebook_name>.ipynb
   ```

### Visualizations
Visual results can be found in the `static_figure/` and `bandit_figure/` directories.

---

## Contributing
If you'd like to contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description of changes.

---

## Contact
For questions or issues, contact Xinyu Fei at xinyuf@umich.edu.

