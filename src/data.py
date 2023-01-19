
import os
import json
import pandas as pd

from PIL import Image
from datasets import Dataset
from datasets import load_dataset as datasets_load_dataset

from torch.utils.data import DataLoader


def ImageCaptioningDataLoader(image_path, label_path, feature_extractor, tokenizer, batch_size, mode="train"):
    """
    Build Data Loader

    """
    def preprocess_function(examples):
        images = [Image.open(os.path.join(image_path, inp+".jpg")).convert("RGB").resize((224,224)) for inp in examples["input"]]
        pixel_values = feature_extractor(images, return_tensors="pt").pixel_values
        if mode=="train":
            tokenizer_input = tokenizer([tokenizer.bos_token+s+tokenizer.eos_token for s in examples["output"]],
                                        padding="max_length", max_length=512, truncation=True, return_tensors="pt", return_token_type_ids=False)
            decoder_input_ids = tokenizer_input["input_ids"]
            decoder_attention_mask = tokenizer_input["attention_mask"]

            return {
                "pixel_values": pixel_values,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
            }
        
        return {
            "pixel_values": pixel_values
        }

    dataset = load_dataset(label_path, mode=mode)
    datasets_load_dataset
    dataset = dataset.map(
        preprocess_function, batched=True, num_proc=8, remove_columns=dataset.column_names
    ).with_format("torch")

    dataloader = DataLoader(dataset, shuffle=(True if mode=="train" else False), batch_size=batch_size)

    return dataloader


def jsonlload(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        j_list = [json.loads(line) for line in lines]

    return j_list


def jsonldump(j_list, fname):
    with open(fname, "w", encoding='utf-8') as f:
        for json_data in j_list:
            f.write(json.dumps(json_data, ensure_ascii=False)+'\n')


def jsonl2df(j_list, mode):
    data_dict = {"input": []}
    if mode=="train":
        data_dict["output"] = []
        for j in j_list:
            for caption in j["output"]:
                data_dict["input"].append(j["input"])
                data_dict["output"].append(caption)
    else:
        for j in j_list:
            data_dict["input"].append(j["input"])
    
    df = pd.DataFrame(data_dict)
    return df


def load_dataset(fname, mode):
    j_list = jsonlload(fname)
    df = jsonl2df(j_list, mode)
    dataset = Dataset.from_pandas(df)

    return dataset