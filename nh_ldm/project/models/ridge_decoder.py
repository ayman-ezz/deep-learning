import numpy as np
import torch
from tqdm import tqdm
import logging
import pickle
import os

logger = logging.getLogger(__name__)

class RidgeRegression:
    """
    Ridge Regression with efficient alpha tuning validation.
    Solves W = (X^T X + alpha * I)^-1 X^T Y
    """
    def __init__(self, alphas=[1.0], fit_intercept=True):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.best_alphas_ = None
        self.mean_X_ = None
        self.mean_Y_ = None

    def fit(self, X, Y, X_val=None, Y_val=None):
        """
        Fit model using closed-form solution.
        If X_val and Y_val are provided, performs validation to select best alpha per target.
        """
        # Convert to numpy if tensors
        if isinstance(X, torch.Tensor): X = X.cpu().numpy()
        if isinstance(Y, torch.Tensor): Y = Y.cpu().numpy()
        if X_val is not None and isinstance(X_val, torch.Tensor): X_val = X_val.cpu().numpy()
        if Y_val is not None and isinstance(Y_val, torch.Tensor): Y_val = Y_val.cpu().numpy()

        n_samples, n_features = X.shape
        n_targets = Y.shape[1]

        # Centering
        if self.fit_intercept:
            self.mean_X_ = np.mean(X, axis=0)
            self.mean_Y_ = np.mean(Y, axis=0)
            X = X - self.mean_X_
            Y = Y - self.mean_Y_
            if X_val is not None:
                X_val = X_val - self.mean_X_
                Y_val = Y_val - self.mean_Y_

        logger.info(f"Computing XtX (shape: {n_features}x{n_features})...")
        XtX = X.T @ X
        XtY = X.T @ Y
        
        best_weights = np.zeros((n_features, n_targets))
        best_scores = np.full(n_targets, -np.inf)
        self.best_alphas_ = np.zeros(n_targets)

        # Iterate over alphas
        for alpha in self.alphas:
            alpha = float(alpha)
            logger.info(f"Solving for alpha={alpha}...")
            # Regularization matrix
            reg = alpha * np.eye(n_features)
            
            # W = (XtX + alpha*I)^-1 XtY
            # Solve (XtX + reg) W = XtY
            try:
                # linalg.solve is generally faster and more stable than inv
                W = np.linalg.solve(XtX + reg, XtY)
            except np.linalg.LinAlgError:
                logger.warning(f"Singular matrix for alpha={alpha}, using pseudoinverse.")
                W = np.linalg.pinv(XtX + reg) @ XtY

            if X_val is not None and Y_val is not None:
                # Evaluate on validation set
                Y_pred_val = X_val @ W
                # Score: R2 or Correlation? Report mentions correlation is typical for fMRI
                # Here we use negative MSE or simple R2 proxy for selection
                # Let's use correlation per target
                
                # Pearson correlation
                # Centered data, so cos sim is practically correlation if normalized
                # But let's be explicit
                denom = (np.linalg.norm(Y_pred_val, axis=0) * np.linalg.norm(Y_val, axis=0))
                denom[denom == 0] = 1e-10
                corrs = np.sum(Y_pred_val * Y_val, axis=0) / denom
                
                # Update best weights
                improved = corrs > best_scores
                best_weights[:, improved] = W[:, improved]
                best_scores[improved] = corrs[improved]
                self.best_alphas_[improved] = alpha
            else:
                # No validation, just use the last alpha or store all?
                # Default to last alpha if no val set provided (shouldn't happen in this pipeline)
                best_weights = W
        
        self.coef_ = best_weights
        logger.info("Fitting complete.")

    def predict(self, X):
        if isinstance(X, torch.Tensor): X = X.cpu().numpy()
        
        if self.fit_intercept and self.mean_X_ is not None:
            X = X - self.mean_X_
            
        pred = X @ self.coef_
        
        if self.fit_intercept and self.mean_Y_ is not None:
            pred = pred + self.mean_Y_
            
        return pred

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

def select_top_voxels(X, Y, keep=5000):
    """
    Selects top voxels based on simple correlation with target mean/PCA or reliability.
    Since we might not have reliability calc here, we can use a simple encoding selection.
    
    Alternative strategy mentioned: "top 5000 most reliable voxels".
    Reliability = correlation between repetitions of same stimulus.
    If unavailable, we select all or use variance.
    """
    # Placeholder for actual reliability selection
    # Assuming X has 15k, we take 5k with highest variance for now if reliability not provided
    vars = np.var(X, axis=0)
    top_indices = np.argsort(vars)[-keep:]
    return np.sort(top_indices)
