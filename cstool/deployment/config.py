import sys
sys.path.append('cstool/deployment')
import torch
from transformer import BertUncased
from transformers import BertTokenizer
from Label2Id import Label2Id


label_tknz = Label2Id()
label_tknz.load_dict("cstool/deployment/model/label_dict/label_dict.pkl")
input_tknz = BertTokenizer.from_pretrained("cstool/deployment/model/input_tknz")

# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = BertUncased()
bert.build_model(len(label_tknz))


# Model Weights Local Path
local_path = "cstool\deployment\model\weights\weights.pth"

# Load Model
bert = BertUncased()
bert.build_model(len(label_tknz))
# bert.load_weights("deployment/model/weights/checkpoint.pth", device=device)
url = 'https://onedrive.live.com/download?cid=5556EBFD8DBB2C25&resid=5556EBFD8DBB2C25%21113&authkey=AHez1aYX36iZDqI'
bert.load_weights_torchhub(url, save_path=local_path)
# hyperparameters for the A.I. model
max_len = 200 