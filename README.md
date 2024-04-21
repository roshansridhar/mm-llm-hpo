# Multimodal Large Language Model Hyperparameter Optimization (MM-LLM-HPO) Framework

## Abstract

This project explores the efficacy of Multimodal Large Language Models (MM-LLMs) in automating Hyperparameter Optimization (HPO) tasks, leveraging their capability to interpret and synthesize both textual and visual data. Through systematic experimentation using the HPOBench dataset, this study compares MM-LLMs with traditional Large Language Models (LLMs) and established HPO methods like random search and Bayesian optimization. Our findings demonstrate that MM-LLMs not only enhance the efficiency of the HPO process but also provide deeper insights through their interpretative capabilities, thereby setting new standards for automated machine learning practices.

## Introduction

Hyperparameter Optimization (HPO) is crucial in fine-tuning machine learning models to maximize performance on various datasets. Traditional methods such as random search and Bayesian optimization, while effective, often do not leverage the latest advancements in AI, particularly in processing complex multimodal data. This project introduces a novel approach using MM-LLMs, which are hypothesized to outperform both single-modal LLMs and traditional methods by integrating insights from both textual descriptions and visual data representations. This approach aims to automate HPO tasks more effectively, reducing both computational costs and manual effort.

## Project Structure
- **/optimizers/**: Contains implementations for various optimizers including random search, Bayesian optimization, LLM-based, and MM-LLM-based optimizers.
- **/benchmarkers/**: Implements the interface to HPOBench, allowing for systematic and reproducible benchmarking across different HPO methods.
- **/utils/**: Utility scripts including configuration management and logging.
- **/examples/**: Example scripts demonstrating the use of the MM-LLM-HPO framework.
- **requirements.txt**: Lists all dependencies required by the project.
- **README.md**: Provides a comprehensive overview of the project, setup instructions, and usage examples.

## Installation Guide for MM-LLM-HPO Framework

This guide walks you through setting up the MM-LLM-HPO framework on your local machine.

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) must be installed on your machine.
- [Swig](https://www.swig.org) to use [SMAC3](https://github.com/automl/SMAC3) bayesian optimization library.

### Environment setup

1. **Create and Activate Conda Environment**  
   Create a new Conda environment with Python 3.9 and activate it:

   ```bash
   conda create --name mm-llm-hpo python=3.9
   conda activate mm-llm-hpo
   ```

2. **Clone and Setup HPOBench**  
Clone the HPOBench repository and install it:
   ```bash
   git clone https://github.com/automl/HPOBench.git
   cd HPOBench
   pip install .
   cd ..
   ```
   
3. **Install Additional Requirements**  
Install swig  
   ```bash
   sudo apt-get install swig (on linux ubuntu)
   -or-
   brew install swig (on mac)
   ```
   Install other necessary Python packages from a requirements file:  
   ```bash
   pip install -r requirements.txt
   ```

### Usage
Navigate to the project directory and run the desired scripts to perform HPO:
```bash
python examples/test1.py
```

# Contributing

Contributions to the MM-LLM-HPO framework are welcome. Please refer to the contributing guidelines for more details on submitting pull requests, reporting bugs, or requesting new features.

# License

This project is licensed under the MIT License.

# Citation

If you use this framework or the findings from our associated paper in your research, please cite it as follows:  
TBD