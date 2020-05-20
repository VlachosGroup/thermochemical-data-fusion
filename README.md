# thermochemical-data-fusion

List of files
=============

<br>1) <b>smiles_to_graphs.py</b>
<br>&emsp;    This file contains code to convert a list of smiles (read from a text file) to the data matrix.
<br>2) <b>lasso_fits.py</b>
<br>&emsp;    This file contains the LASSO implementation used in the paper.
<br>3) <b>bootstrap_error_estimation.py</b>
<br>&emsp;    This file can be used to generate a bootstrap sample of mean absolute error (MAE) for the model.
<br>4) <b>helper_files.py</b>
<br>&emsp;    This file contains a list of plotting helper functions used in the paper.
<br>5) <b>model_details.json</b>
<br>&emsp;    JSON file contains details of the three models (E-E, E-H and E-G) in the following pseudocode format:
<br>&emsp;'electronic energy': {
<br>&emsp;&emsp;	'coeff': list(ee_model.model_.coef_),
<br>&emsp;&emsp;	'X_mean': list(ee_model.X_scaler.mean_),
<br>&emsp;&emsp;	'X_std': list(ee_model.X_scaler.scale_),
<br>&emsp;&emsp;	'y_mean': list(ee_model.y_scaler.mean_),
<br>&emsp;&emsp;	'y_std': list(ee_model.y_scaler.scale_)},
<br>
<br>&emsp;'enthalpy': {
<br>&emsp;&emsp;	'coeff': list(eh_model.model_.coef_),
<br>&emsp;&emsp;	'X_mean': list(eh_model.X_scaler.mean_),
<br>&emsp;&emsp;	'X_std': list(eh_model.X_scaler.scale_),
<br>&emsp;&emsp;	'y_mean': list(eh_model.y_scaler.mean_),
<br>&emsp;&emsp;	'y_std': list(eh_model.y_scaler.scale_)},
<br>
<br>&emsp;'free energy': 	{
<br>&emsp;&emsp;	'coeff': list(eg_model.model_.coef_),
<br>&emsp;&emsp;	'X_mean': list(eg_model.X_scaler.mean_),
<br>&emsp;&emsp;	'X_std': list(eg_model.X_scaler.scale_),
<br>&emsp;&emsp;	'y_mean': list(eg_model.y_scaler.mean_),
<br>&emsp;&emsp;	'y_std': list(eg_model.y_scaler.scale_)}
<br>6) <b>train_mols.txt and test_mols.txt</b>
<br>&emsp;    Ids of train and test molecules. Nomenlature follows the indexing used in QM9.

