import argparse
import os
import re
import numpy as np
import torch
from torch.nn.utils import *
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
# from accelerate import Accelerator, DistributedDataParallelKwargs
# from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import random
from tqdm import tqdm
from GFlowNets import GFlowNet
from MLP import MLP

torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import wandb
from datetime import datetime
# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

seed = 2549

# Determine if training on a GPU or CPU for reproducibility
if torch.cuda.is_available():
    # Make models reproducible on GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # Set the PYTHONHASHSEED environment variable to the chosen seed to make hash-based operations predictable
    np.random.seed(seed) # Set NumPy's random seed to ensure reproducibility of operations using NumPy's random number generator
    random.seed(seed) # Set Python's built-in random module's seed to ensure reproducibility of random operations using Python's random functions
    np.random.seed(seed)
    torch.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the CPU
    torch.cuda.manual_seed(seed) # Set the seed for generating random numbers in PyTorch to ensure reproducibility on the GPU
    torch.cuda.manual_seed_all(seed) # Ensure reproducibility for all GPUs by setting the seed for generating random numbers for all CUDA devices
    torch.backends.cudnn.deterministic = True # Force cuDNN to use only deterministic convolutional algorithms (can slow down computations but guarantees reproducibility)
    torch.backends.cudnn.benchmark = False # Prevent cuDnn from using any algorithms that are nondeterministic
    torch.set_float32_matmul_precision('medium')
    accelerator = "gpu"
    num_devices = torch.cuda.device_count()  # Use all available GPUs
    print(f"사용 가능한 GPU 개수: {num_devices}")
    strategy = "ddp" if num_devices > 1 else None  # Use DDP if multiple GPUs
    
    # Determine if training via DDP or single GPU
    if num_devices > 1:
        from GFlowNets import ProtDataModuleESM2_DDP as ProtDataModuleESM2
    else:
        from GFlowNets import ProtDataModuleESM2
        print('Running on single GPU, using alternative dataloader')
    print(f"Accelerator: {accelerator}, Number of devices: {num_devices}, Strategy: {strategy}")

else:
    # fix random seeds for reproducibility on CPU
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    accelerator = "cpu"
    max_threads = 16
    num_threads = min(os.cpu_count(), max_threads)  # Use all available CPUs up to a maximum of 16
    torch.set_num_threads(num_threads)  # Set the number of threads for PyTorch
    num_devices = 1  # Use the CPU
    strategy = None
    from GFlowNets import ProtDataModuleESM2
    print(f"Accelerator: {accelerator}, Number of threads: {num_threads}, Strategy: {strategy}")


def set_environment_variables():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    os.environ['HF_HOME'] = '/home/elicer/Bio/RLXF/.cache/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/home/elicer/Bio/RLXF/.cache/huggingface/transformers'
    os.environ['HF_DATASETS_CACHE'] = '/home/elicer/Bio/RLXF/.cache/huggingface/datasets'
    os.environ["TORCH_HOME"] = "/home/ec2-user/SageMaker/huggingface_cache/torch"
    os.environ["TMPDIR"] = "/home/ec2-user/SageMaker/tmp"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    os.environ["TORCH_NCCL_TIMEOUT_MS"] = "12000000"
    os.environ["TORCH_TIMEOUT_MS"] = "12000000"
    os.environ["NCCL_TIMEOUT"] = "12000000"
    os.environ["TORCH_NCCL_TIMEOUT"] = "12000000"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    os.environ["NCCL_P2P_DISABLE"] = "0"

def load_model_and_tokenizer(model_name, num_reward_models):
    rl_updated_model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 메모리 절약
        # device_map="auto",  # 자동 device mapping
        # low_cpu_mem_usage=True,  # CPU 메모리 절약
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"

    # load ensemble of reward models
    reward_models = []
    for i in range(num_reward_models):
        reward_model_name = f"reward_model_v{i}.ckpt"
        checkpoint_path = f"./reward_models/{reward_model_name}"
        reward_model = MLP.load_from_checkpoint(checkpoint_path, map_location='cpu')
        reward_model.eval()
        for param in reward_model.parameters():
            param.requires_grad = False
        reward_models.append(reward_model)

    return rl_updated_model, tokenizer, reward_models


def mini_train(prot_type, batch_size, epoch, model_identifier, model_name, num_reward_models, saved_model_path, save_every_n_epochs):
    rl_updated_model, tokenizer, reward_model = load_model_and_tokenizer(model_name, num_reward_models)

    if prot_type == 'CreiLOV':
         WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA'
         predicted_WT_fitness = 4.1498 # Predicted WT score
    elif prot_type == 'avgfp':
         WT = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK' # parent sequence
   
    # accelerator = Accelerator(
    #     mixed_precision="bf16", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    # )

    run_group_name = f"{prot_type}_{current_date}_GFlowNets_training"
    csv_logger = CSVLogger('logs', name=f"GFlowNets_{model_identifier}")
    wandb_logger = WandbLogger(
            project="RLXF",
            name=f"GFlowNets_{model_identifier}_{type}_{num_reward_models}_reward_models_{epoch}_epochs_{seed}_seed",
            group=run_group_name,
            config={
                "model_parameters" : {
                    "model_identifier": model_identifier,
                },
                "GFlowNets_parameters": {
                    "seed": seed,
                    "epochs": epoch,
                    # "iterations": iterations,
                    # "rel_to_WT": rel_to_WT,
                    # "epsilon": epsilon
                },
                # "reward_hyperparameters": {
                #     "pairwise_hd_aver_factor": pairwise_hd_aver_factor,
                #     "dkl_scale_init": dkl_scale_init,
                #     "dkl_scale": dkl_scale
                # },
                # "optimizer_hyperparameters": {
                #     "learning_rate": learning_rate,
                #     "lr_mult": lr_mult,
                #     "lr_mult_factor": lr_mult_factor,
                #     "WD": WD,
                #     "grad_clip_threshold": grad_clip_threshold,
                #     "grad_clip_threshold_factor": grad_clip_threshold_factor
                # },
                # "generation_parameters": {
                #     "num_designs": num_designs,
                #     "num_muts": num_muts,
                #     "high_conf_threshold": high_conf_threshold,
                #     "cum_prob_threshold": cum_prob_threshold,
                #     "ep": ep,
                #     "generation_seed": generation_seed,
                #     "predicted_wt_score": predicted_wt_score
                # },
                "num_reward_models": num_reward_models,
                "sequence_len": len(WT),
                "type": type,
                "WT": WT},
        )
    logger = [csv_logger, wandb_logger]
    version = csv_logger.version

    dm = ProtDataModuleESM2(WT, batch_size=1, seed=seed) # Loading WT to dataloader, we generate variant designs each batch so only load WT initially to models
    
    model = GFlowNet(
        rl_updated_model=rl_updated_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        accelerator=accelerator,
        WT=WT,
        predicted_WT_fitness=predicted_WT_fitness,
        save_every_n_epochs=save_every_n_epochs,
    )

    if strategy == "ddp":
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=epoch,
            # precision=16 if accelerator == "gpu" else 32,  # Mixed precision only on GPU
            precision="bf16-mixed",  # bfloat16 혼합 정밀도
            strategy=DDPStrategy(
                find_unused_parameters=True,  # Accelerator의 find_unused_parameters와 동일
                gradient_as_bucket_view=True,  # 성능 최적화
            ),
            enable_progress_bar=True,
            log_every_n_steps=1,
            accelerator=accelerator,
            num_nodes=1,
            devices=num_devices,
            # strategy=strategy,
            )  
    else:
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=epoch,
            # precision=16 if accelerator == "gpu" else 32,  # Mixed precision only on GPU
            precision="bf16-mixed",  # bfloat16 혼합 정밀도
            strategy=DDPStrategy(
                find_unused_parameters=True,  # Accelerator의 find_unused_parameters와 동일
                gradient_as_bucket_view=True,  # 성능 최적화
            ),
            enable_progress_bar=True,
            log_every_n_steps=1,
            accelerator=accelerator,
            num_nodes=1,
            devices=num_devices,
            )
    
    trainer.fit(model, dm)
    wandb.finish()  # Finish the wandb run for this model

    # for iteration in range(epoch):
    #     for k, batch in enumerate(tqdm(gfn.dataloader), start=1):
    #         with accelerator.accumulate(gfn.model):
    #             protein_sequences = [
    #                 [
    #                     text for text in batch[column]
    #                 ]
    #             ] * gfn.number_generation

    #             ground_truth_answers = [
    #                 extract_answer_gsm8k(text) if choice == 0 else extract_answer_nvidia(text) for text in batch[response]
    #             ]

    #             tokenized_inputs = tokenizer.batch_encode_plus(
    #                 messages, return_tensors="pt", padding="max_length", max_length=1010, padding_side="left"
    #             )

    #             if "token_type_ids" in tokenized_inputs: #needed for some models
    #                 tokenized_inputs.pop("token_type_ids")

    #             gfn.model.module.gradient_checkpointing_disable()
    #             gfn.generate(tokenized_inputs, ground_truth_answers)
    #             gfn.model.module.gradient_checkpointing_enable()

    #             loss, reward = gfn.step()

    #             if k % 200 == 0:
    #                 accelerator.print(f"Iteration {k}")

    #     accelerator.print(f"=========== End of epoch {iteration + 1} ===========")

    # gfn.save_model(saved_model_path)
    # accelerator.print("Model saved successfully!")


def main():
    model_identifier ='esm2_t33_650M_UR50D'
    model_name = f"facebook/{model_identifier}"
    num_reward_models = 2
    filepath = 'GFlowNets'

    parser = argparse.ArgumentParser(description="Train a GFlowNet model.")
    # parser.add_argument("--choice", type=int, default=1, help="Dataset choice: 0 for gsm8k, 1 for nvidia openmath")
    parser.add_argument("--prot_type", type=str, default="CreiLOV", help="Dataset choice: Type of the protein")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=1, help="Number of epochs")
    # parser.add_argument("--model_name", type=str, required=True, help="Path to the model")
    # parser.add_argument("--reward_model_name", type=str, required=True, help="Path to the reward model")
    parser.add_argument("--saved_model_path", type=str, default=f"./logs/{filepath}_{model_identifier}", help="Path to the trained model")

    args = parser.parse_args()
    
    set_environment_variables()
    # mini_train(args.choice, args.batch_size, args.epoch, args.model_name, args.reward_model_name, args.saved_model_path)
    save_every_n_epochs = args.epoch
    mini_train(args.prot_type, args.batch_size, args.epoch, model_identifier, model_name, num_reward_models, args.saved_model_path, save_every_n_epochs)

if __name__ == "__main__":
    main()