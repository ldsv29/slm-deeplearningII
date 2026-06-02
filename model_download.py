from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

load_dotenv() 

snapshot_download(repo_id="ldsv29/slm-pretrained", local_dir="checkpoints", token=os.getenv("HUGGINFACE_TOKEN"))
