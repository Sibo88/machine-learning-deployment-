import pandas as pd
from flask import Flask, request, jsonify
import joblib
import numpy as np
import gdown
import os

# -------------------
# 1. Download the trained model from Google Drive if not exists
# -------------------
MODEL_FILE = "rf_model.pkl"
GOOGLE_DRIVE_ID = "1F70nFMXABwclT4_ZMEnxdpF_3LvdsW73"  # your model link ID
MODEL_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"

if not os.path.exists(MODEL_FILE):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

# -------------------
# 2. Load the trained model and preprocessor
# -------------------
model = joblib.load(MODEL_FILE)
preprocessor = joblib.load('/content/preprocessor.pkl')  # keep your existing preprocessor

# -------------------
# 3. Columns for preprocessing
# -------------------
numerical_cols_final = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'last_fico_range_high', 'last_fico_range_low', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit']
categorical_cols_final = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'pymnt_plan', 'purpose', 'addr_state', 'initial_list_status', 'application_type', 'disbursement_method', 'debt_settlement_flag']

# -------------------
# 4. Initialize Flask app
# -------------------
app = Flask(__name__)

@app.route('/')
def home():
    return 'Credit Risk Prediction API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json(force=True)
        if not json_data:
            return jsonify({'error': 'No data provided'}), 400

        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([json_data])
        
        # Ensure all columns expected by the preprocessor are present
        all_expected_cols = numerical_cols_final + categorical_cols_final
        processed_input_df = pd.DataFrame(columns=all_expected_cols)
        processed_input_df = pd.concat([processed_input_df, input_df], ignore_index=True)
        
        # Fill missing values
        for col in numerical_cols_final:
            if col not in processed_input_df.columns:
                processed_input_df[col] = np.nan
            processed_input_df[col] = pd.to_numeric(processed_input_df[col], errors='coerce')
        for col in categorical_cols_final:
            if col not in processed_input_df.columns:
                processed_input_df[col] = 'unknown'
        
        # Preprocess the input
        input_processed = preprocessor.transform(processed_input_df[all_expected_cols])
        
        # Make prediction
        prediction = model.predict(input_processed)
        prediction_proba = model.predict_proba(input_processed)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_no_default': float(prediction_proba[0][0]),
            'probability_default': float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)