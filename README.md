# thermochemical-data-fusion

List of files
=============

1) smiles_to_graphs.py
    This file contains code to convert a list of smiles (read from a text file) to the data matrix.
2) lasso_fits.py
    This file contains the LASSO implementation used in the paper.
3) bootstrap_error_estimation.py
    This file can be used to generate a bootstrap sample of mean absolute error (MAE) for the model.
4) helper_files.py
    This file contains a list of plotting helper functions used in the paper.
5) 
    JSON file contains details of the three models (E-E, E-H and E-G) in the following pseudocode format:

'electronic energy': {
	'coeff': list(ee_model.model_.coef_),
	'X_mean': list(ee_model.X_scaler.mean_),
	'X_std': list(ee_model.X_scaler.scale_),
	'y_mean': list(ee_model.y_scaler.mean_),
	'y_std': list(ee_model.y_scaler.scale_)},

'enthalpy': {
	'coeff': list(eh_model.model_.coef_),
	'X_mean': list(eh_model.X_scaler.mean_),
	'X_std': list(eh_model.X_scaler.scale_),
	'y_mean': list(eh_model.y_scaler.mean_),
	'y_std': list(eh_model.y_scaler.scale_)},

'free energy': 	{
	'coeff': list(eg_model.model_.coef_),
	'X_mean': list(eg_model.X_scaler.mean_),
	'X_std': list(eg_model.X_scaler.scale_),
	'y_mean': list(eg_model.y_scaler.mean_),
	'y_std': list(eg_model.y_scaler.scale_)}
