
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoFeatureExtractor, VisionEncoderDecoderModel, AutoTokenizer

from src.data import ImageCaptioningDataLoader
from src.module import ImageCaptioningModule
from src.utils import get_logger

# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Train Image Captioning with BART")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output-dir", type=str, required=True, help="output directory path to save artifacts")
g.add_argument("--model-path", type=str, help="model file path")
# g.add_argument("--tokenizer", type=str, required=True, help="huggingface tokenizer path")
g.add_argument("--batch-size", type=int, default=32, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=64, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help=" the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=10, help="the numnber of training epochs")
g.add_argument("--max-learning-rate", type=float, default=2e-4, help="max learning rate")
g.add_argument("--min-learning-rate", type=float, default=1e-5, help="min Learning rate")
g.add_argument("--warmup-rate", type=float, default=0.1, help="warmup step rate")
g.add_argument("--gpus", type=int, default=0, help="the number of gpus")
g.add_argument("--logging-interval", type=int, default=100, help="logging interval")
g.add_argument("--evaluate-interval", type=int, default=500, help="validation interval")
g.add_argument("--seed", type=int, default=42, help="random seed")

g = parser.add_argument_group("Wandb Options")
g.add_argument("--wandb-run-name", type=str, help="wanDB run name")
g.add_argument("--wandb-entity", type=str, help="wanDB entity name")
g.add_argument("--wandb-project", type=str, help="wanDB project name")
# fmt: on


def main(args):
    logger = get_logger("train")

    os.makedirs(args.output_dir)
    logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed)

    logger.info(f"[+] GPU: {args.gpus}")

    logger.info(f"[+] Load Feature Extractor")
    feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    logger.info(f'[+] Load Tokenizer')
    decoder_tokenizer = AutoTokenizer.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token='</s>',
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',
        mask_token='<mask>'
    )

    logger.info(f'[+] Load Dataset')
    train_dataloader = ImageCaptioningDataLoader("/data/data/captioning/" ,"resource/data/nikluge-2022-image-train.jsonl", feature_extractor, decoder_tokenizer, args.batch_size)
    valid_dataloader = ImageCaptioningDataLoader("/data/data/captioning/", "resource/data/nikluge-2022-image-dev.jsonl", feature_extractor, decoder_tokenizer, args.valid_batch_size)
    total_steps = len(train_dataloader) * args.epochs

    if args.model_path:
        logger.info(f'[+] Load Model from "{args.model_path}"')
        model = VisionEncoderDecoderModel.from_pretrained(args.encoder_path, args.decoder_path)
    else:
        logger.info(f'[+] Load Model from "https://huggingface.co/gogamza/kobart-base-v2"')
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k", "skt/kogpt2-base-v2"
        )
        model.config.pad_token_id = decoder_tokenizer.pad_token_id
        
    logger.info(f"[+] Load Pytorch Lightning Module")
    lightning_module = ImageCaptioningModule(
        model,
        feature_extractor,
        total_steps,
        args.max_learning_rate,
        args.min_learning_rate,
        args.warmup_rate,
        args.output_dir
    )

    logger.info(f"[+] Start Training")
    train_loggers = [TensorBoardLogger(args.output_dir, "", "logs")]
    if args.wandb_project:
        train_loggers.append(
            WandbLogger(
                name=args.wandb_run_name or os.path.basename(args.output_dir),
                project=args.wandb_project,
                entity=args.wandb_entity,
                save_dir=args.output_dir,
            )
        )
    
    # If evaluate_interval passed float F, check validation set 1/F times during a training epoch
    if args.evaluate_interval == 1:
        args.evaluate_interval = 1.0
    trainer = pl.Trainer(
        logger=train_loggers,
        max_epochs=args.epochs,
        log_every_n_steps=args.logging_interval,
        val_check_interval=args.evaluate_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        gpus=args.gpus,
    )
    trainer.fit(lightning_module, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
