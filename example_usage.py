import numpy as np
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel 
def main():
    # Example training data.
    X = np.array([[1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5]])
    y = np.array([2.5, 3.5, 4.5, 5.5])
    
    model = LassoHomotopyModel(max_iter=1000, tol=1e-6)
    model.fit(X, y)
    
    intercept = np.mean(y) - np.mean(X @ model.coef_)

    predictions = X @ model.coef_ + intercept  
    print("Predictions:", predictions)
    
    print("Coefficients:", model.coef_)
    print("Intercept:", intercept)

if __name__ == "__main__":
    main()
