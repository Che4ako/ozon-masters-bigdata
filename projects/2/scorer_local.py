#!/opt/conda/envs/dsenv/bin/python

#
# This is a MAE scorer
#
import numpy as np
import sys
import os
import logging
import pandas as pd
from sklearn.metrics import log_loss
#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read true values
#
try:
    true_path, pred_path = sys.argv[1:]
except:
    logging.critical("Parameters: true_path (local) and pred_path (url or local)")
    sys.exit(1)

logging.info(f"TRUE PATH {true_path}")
logging.info(f"PRED PATH {pred_path}")


#open true path
df_true = pd.read_csv(true_path, header=None, index_col=0, names=["id", "true"], sep="\t")

#open pred_path
df_pred = pd.read_csv(pred_path, header=None, index_col=0, names=["id", "pred"], sep=",")

len_true = len(df_true)
len_pred = len(df_pred)

maximum = df_pred['pred'].max()
minimum = df_pred['pred'].min()
number = df_pred['pred'].nunique()
nans = df_pred[['pred']].loc[df_pred['pred'] == np.nan].sum()

shape_true = df_true.shape
shape_pred = df_pred.shape

logging.info(f"TRUE RECORDS {len_true}")
logging.info(f"PRED RECORDS {len_pred}")

logging.info(f"TRUE RECORDS {shape_true}")
logging.info(f"PRED RECORDS {shape_pred}")

logging.info(f"PRED maximum {maximum}")
logging.info(f"PRED minimum {minimum}")
logging.info(f"PRED number {number}")
logging.info(f"PRED nans {nans}")

assert len_true == len_pred, f"Number of records differ in true and predicted sets"

df = df_true.join(df_pred)
len_df = len(df)
assert len_true == len_df, f"Combined true and pred has different number of records: {len_df}"

score = log_loss(df['true'], df['pred'])

print(score)

sys.exit(0)
