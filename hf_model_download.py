# from huggingface_hub import hf_hub_download
# import os

# HUGGING_FACE_API_KEY = "hf_XXXXXXX"

# # Replace this if you want to use a different model
# model_id = "navteca/tapas-large-finetuned-wtq"
# filenames = [
#     "pytorch_model.bin", "config.json",
#     "special_tokens_map.json", "vocab.txt", "tokenizer_config.json"
# ]


# # 	I worked out the filenames by browsing Files and versions on the Hugging Face UI.

# for filename in filenames:
#     downloaded_model_path = hf_hub_download(
#         repo_id=model_id,
#         filename=filename,
#         token=HUGGING_FACE_API_KEY
#     )
#     print(downloaded_model_path)

# print(downloaded_model_path)

import shutil

src_dir = "/Users/ssteni/.cache/huggingface/hub/models--navteca--tapas-large-finetuned-wtq/snapshots/cd7feb8b379e08187f8927debec340fa05ca3715/"
dest_dir = "/Users/ssteni/Documents/tapex/tapas_hf_download"

shutil.copytree(src_dir, dest_dir)
