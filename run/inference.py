
import argparse
from tqdm import tqdm

import torch
from transformers import AutoFeatureExtractor, VisionEncoderDecoderModel, AutoTokenizer

from src.data import ImageCaptioningDataLoader, jsonlload, jsonldump
from src.utils import get_logger


parser = argparse.ArgumentParser(prog="train", description="Inference Image Captioning with BART")

parser.add_argument("--model-ckpt-path", type=str, help="VisionEncoderDecoder model path")
parser.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
parser.add_argument("--output-path", type=str, required=True, help="output jsonl file path")
parser.add_argument("--batch-size", type=int, default=32, help="training batch size")
parser.add_argument("--output-max-seq-len", type=int, default=64, help="output max sequence length")
parser.add_argument("--num-beams", type=int, default=3, help="beam size")
parser.add_argument("--device", type=str, default="cpu", help="inference device")


def main(args):
    logger = get_logger("inference")

    logger.info(f"[+] Use Device: {args.device}")
    device = torch.device(args.device)

    logger.info(f"[+] Load Feature Extractor")
    feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    logger.info(f'[+] Load Tokenizer from "{args.tokenizer}"')
    if args.tokenizer:
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        decoder_tokenizer = AutoTokenizer.from_pretrained(
            "skt/kogpt2-base-v2",
            bos_token='</s>',
            eos_token='</s>',
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>'
        )

    logger.info(f'[+] Load Dataset')
    dataloader = ImageCaptioningDataLoader("/data/data/captioning/", "resource/data/nikluge-2022-image-test.jsonl", feature_extractor, decoder_tokenizer, args.batch_size, mode="infer")

    logger.info(f'[+] Load Model from "{args.model_ckpt_path}"')
    model = VisionEncoderDecoderModel.from_pretrained(args.model_ckpt_path)
    model.to(device)

    logger.info("[+] Eval mode & Disable gradient")
    model.eval()
    torch.set_grad_enabled(False)

    logger.info("[+] Start Inference")
    total_summary_tokens = []
    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        summary_tokens = model.generate(
            pixel_values=pixel_values,
            max_length=args.output_max_seq_len,
            pad_token_id=decoder_tokenizer.pad_token_id,
            bos_token_id=decoder_tokenizer.bos_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id,
            num_beams=args.num_beams,
            use_cache=True,
        )
        total_summary_tokens.extend(summary_tokens.cpu().detach().tolist())

    logger.info("[+] Start Decoding")
    decoded = [decoder_tokenizer.decode(tokens, skip_special_tokens=True) for tokens in tqdm(total_summary_tokens)]
    
    j_list = jsonlload("resource/data/nikluge-2022-image-test.jsonl")
    for idx, oup in enumerate(decoded):
        j_list[idx]["output"] = oup

    jsonldump(j_list, args.output_path)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
