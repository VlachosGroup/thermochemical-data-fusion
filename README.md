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
<br>5) <b>final_model_lasso_EE.json, final_model_lasso_EH.json and final_model_lasso_EG.json</b>
<br>&emsp;    JSON file contains details of the three models (EE, EH and E\G) in the following pseudocode format:
<br>&emsp;'electronic energy': {
<br>&emsp;&emsp;	'coefficients': model_obj.model_.coef_.tolist(),
<br>&emsp;&emsp;	'intercept': model_obj.model_.intercept_,
<br>&emsp;&emsp;	'alpha': model_obj.model_.alpha_,
<br>&emsp;&emsp;	'cv_alphas': model_obj.model_.alphas_.tolist(),
<br>&emsp;&emsp;	'cv_mse_path': model_obj.model_.mse_path_.tolist(),
<br>&emsp;&emsp;	'X_scale_mean': model_obj.X_scaler.mean_.tolist(),
<br>&emsp;&emsp;	'y_scale_mean': float(model_obj.y_scaler.mean_),
<br>&emsp;&emsp;	'X_scaler_std': model_obj.X_scaler.scale_.tolist(),
<br>&emsp;&emsp;	'y_scale_mean': float(model_obj.y_scaler.scale_)
<br>
<br>6) <b>train_mols.txt and test_mols.txt</b>
<br>&emsp;    Ids of train and test molecules. Nomenlature follows the indexing used in QM9.

