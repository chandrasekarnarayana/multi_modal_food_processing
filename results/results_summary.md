# Results Summary

## Training Ablations
| mode          |   best_val_acc |
|:--------------|---------------:|
| fusion        |       0.911111 |
| video_only    |       0.733333 |
| rheology_only |       0.97037  |

## Test Metrics
Test accuracy: 0.9407407407407408

MAE: 0.23011400892592596

RMSE: 0.31134602808942863

## Active Learning (Uncertainty vs Random)
| strategy    |   first_acc |   last_acc |   first_labelled |   last_labelled |
|:------------|------------:|-----------:|-----------------:|----------------:|
| uncertainty |    0.325926 |   0.762963 |               63 |             363 |
| random      |    0.311111 |   0.888889 |               63 |             363 |