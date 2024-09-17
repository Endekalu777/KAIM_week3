# AlphaCare Insurance Solutions (ACIS) - Car Insurance Data Analysis

This project focuses on analyzing historical car insurance claim data to optimize marketing strategies and identify low-risk clients for premium reductions. The analysis includes data visualization, statistical analysis, and model evaluation to support strategic decision-making for AlphaCare Insurance Solutions.

## Project Overview

The core of the project involves using Python scripts and Jupyter notebooks to process, analyze, and visualize insurance claim data. Key tasks include identifying trends, refining pricing models, and leveraging machine learning models to predict insurance premiums.

## Installation

### Creating a Virtual Environment

#### Using Conda

If you prefer Conda as your package manager:

1. Open your terminal or command prompt.
2. Navigate to your project directory.
3. Run the following command to create a new Conda environment:

    ```bash
    conda create --name acis_analysis python=3.12.5
    ```
    - Replace `acis_analysis` with your desired environment name and `3.12.5` with your preferred Python version.

4. Activate the environment:

    ```bash
    conda activate acis_analysis
    ```

#### Using Virtualenv

If you prefer using `venv`, Python's built-in virtual environment module:

1. Open your terminal or command prompt.
2. Navigate to your project directory.
3. Run the following command to create a new virtual environment:

    ```bash
    python -m venv acis_analysis
    ```
    - Replace `acis_analysis` with your desired environment name.

4. Activate the environment:

    - On Windows:
        ```bash
        .\acis_analysis\Scripts\activate
        ```

    - On macOS/Linux:
        ```bash
        source acis_analysis/bin/activate
        ```

### Installing Dependencies with pip

Once your virtual environment is created and activated, install the required dependencies using:

```bash
pip install -r requirements.txt

### Installing Dependencies with Conda

Alternatively, you can use Conda to install the project dependencies. Note that you will need to install each package individually. To do this, first ensure that you have activated your Conda environment, then use the following commands to install each required package:

```bash
conda install -c conda-forge package-name
```

### Clone this package
- To install the network_analysis package, follow these steps:

- Clone the repository:

```bash
git clone https://github.com/your-username/KAIM_week3.git
```
- Navigate to the project directory:

```bash
cd Solar_radiation_analysis
Install the required dependencies:
```

```bash
pip install -r requirements.txt
```


# Data Version Control (DVC)

To manage and version the dataset effectively, DVC (Data Version Control) has been implemented. This ensures reproducibility and efficient tracking of data changes.

## DVC Setup

* **DVC Installation**: DVC was installed to manage data versioning.
* **Storage Setup**: Local storage was configured to track dataset changes.
* **Data Versions**:
  * v1.0: Initial dataset with missing values.
  * v2.0: Cleaned and preprocessed dataset.
* **Remote Setup**: Cloud storage was configured for remote access and collaboration.
* **Pushing Data**: Both versions were pushed to DVC for tracking and retrieval.

## Using DVC

To interact with DVC, use the following commands:

* **Initialize DVC**:
  ```bash
  dvc init

## Add Data

```bash
dvc add data/your_dataset.csv
```

## Commit changes
git add data/your_dataset.csv.dvc .gitignore
git commit -m "Add dataset to DVC"

## Push to remote
```
dvc push
```

## Pull to remote
```
dvc pull
```


## Usage Instructions

Once the dependencies are installed, you can run the analysis notebooks by launching Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

## Contributions
Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

### Contact
For any questions or additional information please contact Endekalu.simon.haile@gmail.com