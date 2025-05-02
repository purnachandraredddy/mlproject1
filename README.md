# Lasso Homotopy Regression

## Spring 2025 Machine Learning (CS-584-04/05)

## Table of Contents
- [Project Overview](#project-overview)
- [Team](#team)
- [Implementation Overview](#implementation-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Implementation Q&A](#implementation-qa)
- [References](#references)
- [License](#license)

---

## Project Overview

This project implements the LASSO (Least Absolute Shrinkage and Selection Operator) regression model using the Homotopy Method, following the approach of Garrigues & El Ghaoui (2008). The implementation provides a robust solution for ℓ₁-regularized least squares problems, supporting both batch and online learning via homotopy continuation. The codebase includes data standardization, active set management, and efficient rank-one matrix updates for numerical stability and computational efficiency.

Key features:
- Efficient solution path computation for LASSO using homotopy continuation.
- Online/streaming data support for sequential updates.
- Visualization tools for coefficient trajectories and model diagnostics.
- Comprehensive test suite for correctness and benchmarking.

---

Team Members
-Purnachandra Reddy Peddasura (A20544751)
- Sudireddy Raghavender Reddy (A20554654)
- Chaitanya Durgesh Nynavarapu (A20561894)
- Jeswanth Jayavarapu (A20547505)
## Implementation Overview

The core algorithm is implemented in `LassoHomotopy/model/LassoHomotopy.py` as the `LassoHomotopyModel` class. It supports:
- Batch initialization and fitting.
- Online updates with new data.
- Regularization path tracking.
- Data standardization and intercept handling.
- Efficient active set and transition point management.

---

## Installation

1. **Clone the repository** and navigate to the project directory.

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv virtenv
   # On macOS/Linux:
   source virtenv/bin/activate
   # On Windows:
   virtenv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Basic Example

```python
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel
import numpy as np

# Load or generate your data
X, y = ...  # shape (n_samples, n_features), (n_samples,)

# Initialize the model
model = LassoHomotopyModel(alpha=0.1, lambda_min=1e-6, max_iter=1000)

# Fit the model
results = model.fit(X, y)

# Predict
predictions = results.predict(X)


```


---

## Testing

A comprehensive test suite is provided in `LassoHomotopy/tests/test_LassoHomotopy.py`.

**To run all tests:**
```bash
cd LassoHomotopy
pytest tests/
```
or
```bash
pytest LassoHomotopy/tests/test_LassoHomotopy.py -v
```

**To compare test results:**
```bash
python LassoHomotopy/tests/compare_test_results.py -v
```

Test outputs and visualizations (e.g., coefficient paths, prediction plots) are saved as `.png` files in the project directory.

---

## Implementation Q&A

**1. What does the model do and when should it be used?**  
This implementation solves ℓ₁-regularized least squares (LASSO) problems using homotopy continuation, providing exact solution paths as the regularization parameter varies. It is ideal for:
- Feature selection in high-dimensional data
- Streaming/online learning scenarios
- Problems requiring frequent model updates
- Situations where model interpretability and sparsity are important

**2. How did you test your model?**  
- Comparison against scikit-learn's Lasso for numerical correctness
- Recovery of known sparse signals in synthetic data
- Verification of KKT optimality conditions
- Visual diagnostics (plots of predictions, coefficients, residuals)
- Online update and batch processing tests

**3. What parameters can users tune?**  
- `alpha`: Regularization strength (higher = more sparsity)
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance
- `lambda_min`: lambda min parameter

**4. Are there specific inputs your implementation struggles with?**  
- Highly correlated features may cause unstable paths (mitigate with preprocessing or higher λ)
- Very large or dense datasets may slow down updates
- Extremely small λ values may revert to OLS solution due to numerical precision

---

## References

- Garrigues, P. J., & El Ghaoui, L. (2008). An Homotopy Algorithm for the Lasso with Online Observations. *Advances in Neural Information Processing Systems*.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society*.
- Efron, B., et al. (2004). Least angle regression. *Annals of Statistics*.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

*For any questions, please contact the project maintainers.*