import numpy as np
import matplotlib.pyplot as plt

class LassoHomotopyModel:
    def __init__(self, alpha=0.01, max_iter=1000, tol=1e-6, lambda_min=1e-6):  # Reduced alpha and increased precision
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_max = None
        self.lambda_min = lambda_min
        self.coef_ = None
        self.X_mean_ = None
        self.X_std_ = None
        self.y_mean_ = None
        self.lambda_path = []
        self.coef_path = []
        self.active_set = set()
        self.v = None
        self.X1 = None
        self.xn1 = None
        self.theta1 = None
        self.t0 = 0

    def fit(self, X, y):
        """Fit the Lasso model using the RecLasso homotopy algorithm."""
        n_samples, n_features = X.shape
        
        # Standardize the data with robust scaling
        self.X_mean_ = np.mean(X, axis=0)
        self.X_std_ = np.std(X, axis=0) + 1e-10  # Avoid exact zeros
        self.y_mean_ = np.mean(y)
        
        X_scaled = (X - self.X_mean_) / self.X_std_
        y_scaled = y - self.y_mean_
        
        # Initialize coefficients and sign vector
        self.coef_ = np.zeros(n_features)
        self.v = np.zeros(n_features)
        
        # Initialize with strongest correlations
        correlations = np.abs(X_scaled.T @ y_scaled)
        sorted_idx = np.argsort(-correlations)  # Sort in descending order
        initial_features = sorted_idx[:min(3, n_features)]  # Take top 3 features
        
        self.active_set = set(initial_features)
        for idx in initial_features:
            self.v[idx] = np.sign(X_scaled[:, idx].T @ y_scaled)
        
        # Compute initial coefficients for active set
        X_active = X_scaled[:, list(self.active_set)]
        reg_matrix = self.tol * np.eye(len(self.active_set))
        initial_coef = np.linalg.inv(X_active.T @ X_active + reg_matrix) @ X_active.T @ y_scaled
        
        for i, idx in enumerate(self.active_set):
            self.coef_[idx] = initial_coef[i]
        
        # For each sample, follow Algorithm 1 with improved stability
        for n in range(1, n_samples):
            self._compute_path(X_scaled[:n], y_scaled[:n], X_scaled[n], y_scaled[n])
            
            # Refine solution with coordinate descent
            self._refine_solution(X_scaled[:n+1], y_scaled[:n+1])
            
    def _compute_path(self, X_prev, y_prev, xn, yn):
        """Compute the path from θ(n) to θ(0, μn+1)."""
        if len(self.active_set) == 0:
            correlations = np.abs(X_prev.T @ y_prev)
            max_corr_idx = np.argmax(correlations)
            self.active_set = {max_corr_idx}
            self.v[max_corr_idx] = np.sign(X_prev[:, max_corr_idx].T @ y_prev)
        
        self._update_active_set_matrices(X_prev, xn)
        
        if len(self.active_set) > 0:
            active_indices = list(self.active_set)
            reg_matrix = self.tol * np.eye(len(active_indices))
            self.theta1 = np.linalg.inv(self.X1.T @ self.X1 + reg_matrix) @ (
                self.X1.T @ y_prev - self.alpha * self.v[active_indices]
            )
        
        max_iter = 100
        iter_count = 0
        while iter_count < max_iter:
            t0_new = self._compute_next_transition_point(X_prev)
            
            if t0_new <= self.t0 or t0_new > 1:
                break
                
            self.t0 = t0_new
            self._update_active_set(t0_new, X_prev)
            self._update_active_set_matrices(X_prev, xn)
            
            if len(self.active_set) > 0:
                active_indices = list(self.active_set)
                reg_matrix = self.tol * np.eye(len(active_indices))
                self.theta1 = np.linalg.inv(self.X1.T @ self.X1 + reg_matrix) @ (
                    self.X1.T @ y_prev - self.alpha * self.v[active_indices]
                )
            
            iter_count += 1
        
        self._compute_final_value(X_prev, y_prev, xn, yn)
        
    def _refine_solution(self, X, y):
        """Refine solution using coordinate descent."""
        if len(self.active_set) == 0:
            return
            
        max_iter = 10
        for _ in range(max_iter):
            old_coef = self.coef_.copy()
            
            for j in self.active_set:
                X_j = X[:, j]
                # Compute partial residual
                r_j = y - X @ self.coef_ + X_j * self.coef_[j]
                # Update coefficient
                self.coef_[j] = self.soft_threshold(X_j.T @ r_j, self.alpha) / (X_j.T @ X_j + self.tol)
            
            # Check convergence
            if np.max(np.abs(self.coef_ - old_coef)) < self.tol:
                break
                
    def _update_active_set_matrices(self, X_prev, xn):
        """Update matrices for active set."""
        if len(self.active_set) == 0:
            self.X1 = None
            self.xn1 = None
            return
            
        active_indices = list(self.active_set)
        self.X1 = X_prev[:, active_indices]
        self.xn1 = xn[active_indices]
        
    def _compute_next_transition_point(self, X_prev):
        """Compute the next transition point t0."""
        if len(self.active_set) == 0:
            return 1.0
            
        inactive_features = set(range(self.coef_.shape[0])) - self.active_set
        t0_candidates = []
        
        if len(self.active_set) > 0 and self.theta1 is not None:
            for i, j in enumerate(list(self.active_set)):
                if self.theta1[i] != 0:
                    t0 = -self.theta1[i] / (self.xn1[i] - self.theta1[i] + self.tol)
                    if 0 < t0 < 1:
                        t0_candidates.append(t0)
        
        if self.X1 is not None and self.xn1 is not None:
            for j in inactive_features:
                xj = X_prev[:, j]
                w2 = np.abs(xj.T @ self.X1 @ self.xn1 - self.alpha * self.v[j])
                if w2 > 0:
                    t0 = (1 - w2) / (w2 + 1)
                    if 0 < t0 < 1:
                        t0_candidates.append(t0)
                        
        return min(t0_candidates) if t0_candidates else 1.0
        
    def _update_active_set(self, t0, X_prev):
        """Update active set based on transition point."""
        if self.theta1 is not None:
            for i, j in enumerate(list(self.active_set)):
                if self.theta1[i] != 0:
                    t_zero = -self.theta1[i] / (self.xn1[i] - self.theta1[i] + self.tol)
                    if abs(t_zero - t0) < self.tol:
                        self.active_set.remove(j)
                        self.v[j] = 0
                        
        if self.X1 is not None and self.xn1 is not None:
            inactive_features = set(range(self.coef_.shape[0])) - self.active_set
            for j in inactive_features:
                xj = X_prev[:, j]
                w2 = np.abs(xj.T @ self.X1 @ self.xn1 - self.alpha * self.v[j])
                if abs(w2 - 1) < self.tol:
                    self.active_set.add(j)
                    self.v[j] = np.sign(xj.T @ self.X1 @ self.xn1)
                
    def _compute_final_value(self, X_prev, y_prev, xn, yn):
        """Compute final value at t = 1."""
        if len(self.active_set) == 0:
            self.coef_ = np.zeros(self.coef_.shape[0])
            return
            
        self._update_active_set_matrices(X_prev, xn)
        
        active_indices = list(self.active_set)
        reg_matrix = self.tol * np.eye(len(active_indices))
        self.theta1 = np.linalg.inv(self.X1.T @ self.X1 + reg_matrix) @ (
            self.X1.T @ y_prev - self.alpha * self.v[active_indices]
        )
        
        self.coef_ = np.zeros(self.coef_.shape[0])
        for i, j in enumerate(self.active_set):
            self.coef_[j] = self.theta1[i]
            
    def predict(self, X):
        """Make predictions using the fitted model."""
        X_scaled = (X - self.X_mean_) / self.X_std_
        y_pred_scaled = X_scaled @ self.coef_
        return y_pred_scaled + self.y_mean_
        
    def soft_threshold(self, x, lambda_val):
        """Apply soft thresholding operator."""
        return np.sign(x) * np.maximum(np.abs(x) - lambda_val, 0)

    def plot_lambda_path(self):
        """ Plot the coefficient paths as lambda decreases. """
        if not self.lambda_path or not self.coef_path:
            raise ValueError("Model must be fit before plotting lambda path")
            
        plt.figure(figsize=(12, 6))
        coef_path = np.array(self.coef_path)
        
        for j in range(coef_path.shape[1]):
            plt.plot(self.lambda_path, coef_path[:, j], label=f'Feature {j+1}')
            
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('Coefficient Value')
        plt.title('Lasso Coefficient Paths')
        plt.legend()
        plt.grid(True)
        plt.savefig('lasso_path.png')
        plt.close()

    def get_active_features(self):
        """ Return the indices of active features (non-zero coefficients). """
        return list(self.active_set) 