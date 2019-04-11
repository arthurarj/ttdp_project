import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def compute_metrics(y_true, y_pred, y_true_denorm, y_pred_denorm, indices = None):
    
    # Denormalized (real) values should be non-negative since they represent time duration
    if np.sum(y_pred_denorm < 0):
        print('Error: y_pred_denorm is not allowed to have negative values.')
        return 
    
    n_samples, n_models = y_pred_denorm.shape
    
    # Create empty array for individual metrics
    EXPVAR = np.empty(n_models)
    RMSLE  = np.empty(n_models)
    MAPE   = np.empty(n_models)
    MdAPE  = np.empty(n_models)
    MAE   = np.empty(n_models)
    MdAE  = np.empty(n_models)

    # Compute metrics
    for m in range(n_models):
        y_p = y_pred_denorm[:,m].reshape(-1,1)
        EXPVAR[m] = metrics.explained_variance_score(y_true_denorm, y_p)
        RMSLE[m]  = np.sqrt(metrics.mean_squared_log_error(y_true_denorm, y_p))
        MdAPE[m] = np.median(np.abs(y_p - y_true_denorm)/y_true_denorm)
        MAPE[m]  = np.mean(np.abs(y_p - y_true_denorm)/y_true_denorm)
        MAE[m]   = np.mean(np.abs(y_p - y_true_denorm))
        MdAE[m] = np.median(np.abs(y_p - y_true_denorm))

    col = ['Exp. Var.', 'RMSLE', 'MAPE', 'MdAPE', 'MAE', 'MdAE']
    return pd.DataFrame(np.array([EXPVAR, RMSLE, MAPE, MdAPE, MAE, MdAE]).T, columns= col, index = indices)

def plot_acc_window(y_pred, y_true, indices = None):
    n_samples, n_models = y_pred.shape
    tolerance = [2,4,8,10,24,36,48,72,96]

    plt.figure(figsize=(10,6))

    for m in range(n_models):
        y_p = y_pred[:,m].reshape(-1,1)
        acc = [np.mean(np.isclose(y_p,y_true,atol=tol)) for tol in tolerance]
        plt.plot(tolerance, acc, label = None if indices is None else indices[m])
        
    i = 2
    plt.vlines(x=i, ymin=0, ymax=1, color = 'k', linestyles='dashed', alpha=.3)
    plt.text(i+1, .82, str(i) + " hour(s)", rotation=90, verticalalignment='bottom')
    for i in [24,48,72,96]:
        plt.vlines(x=i, ymin=0, ymax=1, color = 'k', linestyles='dashed', alpha=.3)
        plt.text(i+1, .04, str(i//24) + " day(s)", rotation=90, verticalalignment='bottom')
    plt.xlim((0,100))
    plt.ylim((0,1))
    plt.xlabel('Tolerance for Delivery Window (hours)')
    plt.ylabel('"TP" Rate')
    if indices is not None:
        plt.legend(loc=7)
    plt.show()