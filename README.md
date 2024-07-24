# Multilab

 Multilab - A Cleanlab extension for multi-label multi-annotator dataset for exploring label quality, annotator quality and re-labelling images set based on the active learning scores 



## Installation
It is expected that Python (version 3) is installed. Use the package manager pip to install Cleanlab. Cleanlab supports Linux, macOS, and Windows and runs on Python 3.8+.  Please install the other dependencies specified in this [requirements.txt](Multilab/main/requirements.txt) file before running the notebook.
```python
pip install cleanlab
```
To install the package with all optional dependencies:
```python
pip install "cleanlab[all]"
```

## Usage
The `sample.ipynb` includes an example with sample data demonstrating how to use the code for multi-label, multi-annotator tasks. The general input parameters required are:
- In multiannotators.py lines 16 and 18 respectively replace the annotator names and the labels names in the order of your dataset.

- labels_multiannotator : 3D pandas DataFrame or array of multiple given labels per class for each example with shape (N, M, K)  
   
  N is the number of examples, M is the number of annotators. labels_multiannotator[n][m][k] - label for n-th example given by m-th annotator for k-th class.  
 
  For a dataset with K classes, each given label must be binary either 0(absent), 1(present) or NaN if this annotator did not label a particular example.  
        
- pred_probs : np.ndarray
        An array of shape (N, K) of predicted class probabilities from a trained classifier model.

## File Structure
* `main/`
  - `multiannotator_utils.py`
  - `multiannotators.py`
  - `rank.py`
  - `requirements.txt`
- `sample.ipynb`
- `README`

The `multiannotators.py` file contains the primary code for evaluating label quality, annotator quality, and active learning scores for both labeled and unlabeled multi-label datasets. The `multiannotator_utils` and `rank` modules include various helper functions used by `multiannotators.py`.


## Acknowledgements

This project utilizes the Cleanlab code as a foundation, parts of which we have used to support multi-label and multi-annotator data. We appreciate the original work by the Cleanlab team.

The original Cleanlab code can be found at their GitHub repository: [Cleanlab GitHub Repository](https://github.com/cleanlab/cleanlab).

We have adhered to the licensing terms and provided the source for copyrights.
