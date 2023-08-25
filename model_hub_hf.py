# This code can be used to download any model from hugging face using hugging face API key

# install
# pip install 'langchain[llms]' huggingface-hub langchain transformers

from huggingface_hub import hf_hub_download

HUGGING_FACE_API_KEY = "<hugging-face-api-key-goes-here>"

# Replace this if you want to use a different model
model_id = "microsoft/GODEL-v1_1-base-seq2seq"
filenames = [
    "pytorch_model.bin", "added_tokens.json", "config.json", "generation_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"
]

# 	I worked out the filenames by browsing Files and versions on the Hugging Face UI.

for filename in filenames:
    downloaded_model_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        token=HUGGING_FACE_API_KEY
    )

    print(downloaded_model_path)

print(downloaded_model_path)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Running the LLM
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

# And now we’re going to create an instance of our model
model_id = "microsoft/GODEL-v1_1-base-seq2seq"
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text2text-generation",
    model_kwargs={"temperature": 0, "max_length": 1000},
)

