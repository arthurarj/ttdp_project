import numpy as np
import pandas as pd
from sklearn import metrics

def compute_metrics(y_true, y_pred, y_true_denorm, y_pred_denorm, indices):
    
    # Denormalized (real) values should be non-negative since they represent time duration
    if np.sum(y_pred_denorm < 0):
        print('Error: y_pred_denorm is not allowed to have negative values.')
        return 
    
    metric_names = ['explained_variance_score', 'r2_score']

    svm_metrics = []
    for r in range(len(y_pred_denorm)):
        svm_metrics.append([getattr(metrics, metric)(y_true, y_pred[r]) for metric in metric_names])
    svm_metrics = np.array(svm_metrics)

    # MSE and nMSE
    MSE  = np.array([metrics.mean_squared_error(y_true_denorm, y_pred_denorm[r]) for r in range(len(y_pred_denorm))])
    nMSE = np.array([metrics.mean_squared_error(y_true,   y_pred[r]) for r in range(len(y_pred))])

    svm_metrics = np.insert(svm_metrics, svm_metrics.shape[-1], MSE, axis=1)
    svm_metrics = np.insert(svm_metrics, svm_metrics.shape[-1], np.sqrt(MSE), axis=1)
    svm_metrics = np.insert(svm_metrics, svm_metrics.shape[-1], nMSE, axis=1)
    svm_metrics = np.insert(svm_metrics, svm_metrics.shape[-1], np.sqrt(nMSE), axis=1)

    # MSLE and nMSLE
    MSLE  = np.array([metrics.mean_squared_log_error(y_true_denorm, y_pred_denorm[r]) for r in range(len(y_pred_denorm))])

    svm_metrics = np.insert(svm_metrics, svm_metrics.shape[-1], MSLE, axis=1)
    svm_metrics = np.insert(svm_metrics, svm_metrics.shape[-1], np.sqrt(MSLE), axis=1)

    col = ['Exp. Var.', 'R2', 'MSE', 'RMSE', 'nMSE', 'nRMSE', 'MSLE', 'RMSLE']
    svm_df = pd.DataFrame(svm_metrics, columns= col, index = indices)
    
    return svm_df