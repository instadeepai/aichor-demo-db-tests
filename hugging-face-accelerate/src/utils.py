import os
import shutil

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForSequenceClassification
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

from slugify import slugify
from s3fs import S3FileSystem

AWS_ENDPOINT_URL: str = "AWS_ENDPOINT_URL"
AICHOR_INPUT_PATH: str = "AICHOR_INPUT_PATH"
AICHOR_OUTPUT_PATH: str = "AICHOR_OUTPUT_PATH"
AICHOR_OUTPUT_BUCKET_NAME: str = "AICHOR_OUTPUT_BUCKET_NAME"
TENSORBOARD_PATH: str = "AICHOR_LOGS_PATH"

HF_TOKEN: str = "HF_TOKEN"

def get_tokenizer(accelerator: Accelerator, s3: S3FileSystem, model_name: str) -> (PreTrainedTokenizer | PreTrainedTokenizerFast):
    model_slug = f"{slugify(model_name)}-tokenizer"
    local_path = model_slug
    load_from = ""
    should_save_to_s3 = False
    s3_path = os.environ.get(AICHOR_INPUT_PATH) + model_slug

    # download model from S3 if present
    if s3.exists(s3_path):
        # only main process should download from s3
        if accelerator.is_local_main_process:
            s3.get(s3_path, local_path, recursive=True)
        load_from = local_path
    else: # download from HuggingFace
        load_from = model_name
        should_save_to_s3 = True

    accelerator.wait_for_everyone() # wait for local main process to finish downloading the tokenizer from s3
    tokenizer = AutoTokenizer.from_pretrained(load_from, token=os.environ.get(HF_TOKEN))

    accelerator.wait_for_everyone() # wait for all tokenizer loaded on all processes

    # cleanup downloaded model from S3
    if (not should_save_to_s3) and accelerator.is_local_main_process:
        shutil.rmtree(local_path)

    # save downloaded model from HuggingFace to S3
    if should_save_to_s3 and accelerator.is_main_process:
        tokenizer.save_pretrained(local_path)
        s3.put(local_path, s3_path, recursive=True)
        shutil.rmtree(local_path)

    accelerator.wait_for_everyone() # wait cleanup tasks to end
    return tokenizer

def get_model(accelerator: Accelerator, s3: S3FileSystem, model_name: str):
    model_slug = f"{slugify(model_name)}-model"
    local_path = model_slug
    load_from = ""
    should_save_to_s3 = False
    s3_path = os.environ.get(AICHOR_INPUT_PATH) + model_slug

    # download model from S3 if present
    if s3.exists(s3_path):
        # only main process should download from s3
        if accelerator.is_local_main_process:
            s3.get(s3_path, local_path, recursive=True)
        load_from = local_path
    else: # download from HuggingFace
        load_from = model_name
        should_save_to_s3 = True

    accelerator.wait_for_everyone() # wait for local main process to finish downloading the tokenizer from s3
    model = AutoModelForSequenceClassification.from_pretrained(load_from, token=os.environ.get(HF_TOKEN))

    # cleanup downloaded model from S3 from local main process
    if (not should_save_to_s3) and accelerator.is_local_main_process:
        shutil.rmtree(local_path)

    accelerator.wait_for_everyone() # wait for all model loaded on all processes
    if should_save_to_s3 and accelerator.is_main_process:
        model.save_pretrained(local_path)
        s3.put(local_path, s3_path, recursive=True)
        shutil.rmtree(local_path)

    accelerator.wait_for_everyone() # wait for local main process to finish cleaning directory
    return model

def get_dataset(accelerator: Accelerator, s3: S3FileSystem) -> (Dataset | DatasetDict):
    s3_path = os.environ.get(AICHOR_INPUT_PATH) + "glue-mrpc"
    dataset: Dataset | DatasetDict

    if s3.exists(s3_path):
        dataset = load_from_disk(s3_path) # accepts S3 paths
    else:
        dataset = load_dataset("glue", "mrpc")
        if accelerator.is_main_process:
            dataset.save_to_disk(s3_path) # accepts S3 paths
        accelerator.wait_for_everyone()

    return dataset

def save_final_model(accelerator: Accelerator, model, s3: S3FileSystem):
    local_path = "final_model"
    output_path = os.environ.get(AICHOR_OUTPUT_PATH)

    if accelerator.is_main_process:
        print(f"Saving trained model at: {output_path} from main process")
        accelerator.save_model(model, local_path)
        s3.put(local_path, output_path, recursive=True)
        shutil.rmtree(local_path)
        print("Uploaded")

    accelerator.wait_for_everyone()

def save_checkpoint(accelerator: Accelerator, epoch: int, num_epochs: int, checkpoint_dir: str, s3: S3FileSystem):
    epoch = str(epoch).rjust(len(str(num_epochs)), "0")
    output_path = f"s3://{os.environ.get(AICHOR_OUTPUT_BUCKET_NAME)}/{checkpoint_dir}/checkpoint_epoch_{epoch}"
    path = accelerator.save_state()
    if accelerator.is_main_process:
        s3.put(path, output_path, recursive=True)
        # saving a "valid" file to make sure that checkpoint was fully saved.
        with s3.open(f"{output_path}/valid", "w") as f:
            f.write("1")
            f.flush()
        print(f"Checkpoint saved at {output_path}")
    accelerator.wait_for_everyone()
    

def load_checkpoint(accelerator: Accelerator, checkpoint_path: str, s3: S3FileSystem):
    if accelerator.is_local_main_process:
        checkpoint_local_path = "tmp_checkpoint"
        print(f"Loading checkpoint from {checkpoint_path}")
        s3.get(checkpoint_path, checkpoint_local_path, recursive=True)
    accelerator.wait_for_everyone()
    accelerator.load_state(checkpoint_local_path)
    if accelerator.is_local_main_process:
        shutil.rmtree(checkpoint_local_path)

    # get epoch from checkpoint name
    checkpoint_name = checkpoint_path.split('/')[-1]
    epoch = int(checkpoint_name.replace("checkpoint_epoch_", "")) + 1
    return epoch

def get_last_checkpoint_path(checkpoint_dir: str, s3: S3FileSystem):
    checkpoint_dir_full = f"s3://{os.environ.get(AICHOR_OUTPUT_BUCKET_NAME)}/{checkpoint_dir}"
    try:
        dirs = s3.listdir(checkpoint_dir_full)
    except FileNotFoundError:
        print(f"Couldn't find checkpoint at {checkpoint_dir_full}, starting from epoch 0")
        return None
    sorted_dirs = sorted(dirs, key=lambda x: x['Key'], reverse=True)
    for directory in sorted_dirs:
        directory_key = directory['Key']
        files_in_dir = s3.listdir(f"s3://{directory_key}")
        for file in files_in_dir:
            if file['Key'].endswith('/valid'):
                return f"s3://{directory['Key']}"

    return None

