import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from cstool.deployment import config
from transformer import predict

"""
Prediction test run
NOTE: It took my laptop (cuda 11.5, rtx3060 @60W) 8.3 second to make one prediction
"""


def predict_email_content(sentence):
    # NOTE: single prediction
    sent = [sentence]
    # # NOTE: multi predictions
    # sent = [
    #     "Please give me my *Y*&#@*$&( refund",
    #     "OK, thank you",
    #     "Hey, what's the summer program about?"
    # ]
    pred = predict(sent, config.bert.model, config.input_tknz, config.label_tknz, device=config.device)

    return pred