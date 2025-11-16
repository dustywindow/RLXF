import numpy as np
import torch
from torch.nn.utils import *
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, StoppingCriteriaList, StoppingCriteria, BitsAndBytesConfig
# from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb
import re
import random
import itertools
from collections import OrderedDict
from torchtext import vocab # This package can give problems sometimes, it may be necessary to downgrade to a specific version
# from datasets import load_dataset, Dataset
# from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
from tqdm import tqdm
from GFN_utils import ReplayBuffer, get_newline_token_id


# accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])#, dataloader_config=dataloader_config) #,gradient_accumulation_steps=2,
# gradient_checkpointing=True, gradient_checkpointing_enable=True)  
class GFlowNet(pl.LightningModule):
    def __init__(self,
                 rl_updated_model,
                 tokenizer,
                 reward_model,
                 accelerator,
                 WT,
                 predicted_WT_fitness,
                 save_every_n_epochs,
                 pairwise_hd_aver_factor=1.0e-06,
                 learning_rate=3e-2,
                 number_generation=2,
                 subTB_lambda=1.0,
                 temperature=0.6,
                 max_new_tokens=800,
                 num_mutations=5,
                 high_conf_threshold=0.7,
                 cum_prob_threshold=0.1,
                 rel_to_WT=1,
                ):
        super().__init__()

        self.automatic_optimization = False

        self.ReplayBuffer = ReplayBuffer(1000)
        self.subTB_lambda = subTB_lambda
        self.number_generation = number_generation
        self.rl_updated_model = rl_updated_model
        self.rl_updated_model.bfloat16()
        self.rl_updated_model.to('cpu') # Do not need to clear cache. 0 MB freed
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.reward_models = reward_model
        self.learning_rate = learning_rate
        self.alpha = torch.zeros((10000,1), requires_grad=True)

        optimizers_config = self.configure_optimizers()
        self.optimizer = optimizers_config["optimizer"]
        self.scheduler = optimizers_config["lr_scheduler"]

        # self.newline_token_id = get_newline_token_id(self.tokenizer)
        # self.eos_token_id = tokenizer.eos_token_id

        # ESM2 specific parameters
        self.num_mutations = num_mutations
        self.high_conf_threshold = high_conf_threshold
        self.cum_prob_threshold = cum_prob_threshold
        self.rel_to_WT = rel_to_WT

        # reward hyperparameters
        self.pairwise_hd_aver_factor = pairwise_hd_aver_factor

        AAs = 'ACDEFGHIKLMNPQRSTVWY' # setup torchtext vocab to map AAs to indices for reward models
        aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
        aa2ind.set_default_index(20) # set unknown characters to gap
        self.aa2ind = aa2ind

        # self.accelerator = accelerator
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.compteur = 1
        self.avg_loss = 0
        self.avg_reward = 0
        # self.stop_words_ids = [torch.tensor(newline_id) for newline_id in get_newline_token_id(self.tokenizer)]
        self.WT = WT
        self.sequence_len=len(WT)
        self.predicted_WT_fitness = predicted_WT_fitness
        self.save_every_n_epochs = save_every_n_epochs

    def configure_optimizers(self):
        """ Configure optimizers and optionally a scheduler with warm restarts. """
        optimizer = torch.optim.AdamW([
            {'params': self.rl_updated_model.parameters()},
            {'params': self.alpha, 'lr':1e-1}
            
        ], lr=self.learning_rate) 
        # self.num_steps = len(self.dataloader) #// 2
        num_steps = 2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-8)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
    
    def calculate_reward(self, mutated_seqs):
        """Calculate fitness for proteins created by the model"""
        print("Calculating rewards for mutated sequences...")
        self.batch_size = len(mutated_seqs)
        batch_size = self.batch_size
        print(f"Number of mutated sequences: {batch_size}")
        scores_tensor = torch.zeros((len(self.reward_models), batch_size), dtype=torch.float32).to(self.device)
        pre_scores_tensor = torch.zeros((len(self.reward_models), batch_size), dtype=torch.float32).to(self.device)

        # Load all reward models onto the GPU
        for reward_model in self.reward_models:
            reward_model.to(self.device)

        # Compute scores for mutated sequences
        with torch.no_grad():
            for i, reward_model in enumerate(self.reward_models):
                reward_model.eval()  # Set the model to evaluation mode

                for j, seq in enumerate(mutated_seqs):
                    score = reward_model.predict(seq)[0][0]  # Extract score for the sequence from the current model
                    print(f"Score for sequence {j} by model {i}: {score}")
                    scores_tensor[i, j] = score

                # for j, seq in enumerate(pretrained_mutated_seqs):
                #     score = model.predict(seq)[0][0]  # Extract score for the sequence from the current model
                #     print(f"Score for sequence {j} by model {i}: {score}")
                #     pre_scores_tensor[i, j] = score

        # Unload all reward models from the GPU
        for reward_model in self.reward_models:
            reward_model.to('cpu')
        
        # Calculate fitness
        predicted_WT_fitness = self.predicted_WT_fitness # Predicted WT score
        rl_fitness_per_sequence = torch.quantile(scores_tensor, 0.05, dim=0)
        # pre_fitness_per_sequence = torch.quantile(pre_scores_tensor, 0.05, dim=0)
        print(f"RL-updated mean fitness: {rl_fitness_per_sequence.mean()}")
        # print(f"Pre-trained mean fitness: {pre_fitness_per_sequence.mean()}")

        # Compute the overall fitness score based on average_type
        rl_fitness = rl_fitness_per_sequence.max()
        # pre_fitness = pre_fitness_per_sequence.max()
        rel_WT_fitness = rl_fitness / predicted_WT_fitness

        if self.rel_to_WT == 1:
            fitness_advantage = rel_WT_fitness
        else:
            # fitness_advantage = ((rl_fitness - pre_fitness)/pre_fitness)*100
            fitness_advantage = rl_fitness

        self.current_rel_WT_fitness = rel_WT_fitness.item()

        pairwise_hd_aver, total_distance, num_pairs = self.average_pairwise_hamming_distance(mutated_seqs)
        # total_reward = (fitness_advantage + self.pairwise_hd_aver_factor*pairwise_hd_aver - current_beta * dkl_value)
        total_reward = (fitness_advantage + self.pairwise_hd_aver_factor*pairwise_hd_aver)

        # return fitness_advantage, rel_WT_fitness
        return total_reward, rel_WT_fitness

    def sample(self):
        return self.ReplayBuffer.sample_weighted_and_remove(self.batch_size)
        #return self.ReplayBuffer.sample_weighted(self.batch_size)

    # def save_model(self, path):
    #     self.accelerator.wait_for_everyone()
    #     unwrapped_model = self.accelerator.unwrap_model(self.model)
    #     #unwrapped_model = unwrapped_model.merge_and_unload()
    #     unwrapped_model.save_pretrained(path, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save)
    #     self.tokenizer.save_pretrained(path)
    #     self.run.finish()
    #     self.accelerator.wait_for_everyone()

    def calculate_loss_from_replay(self):
        """Calculate GFlowNet loss from replay buffer"""
        loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        sum_rewards = torch.tensor(0.0, device=self.device)
        sum_mean_reward = torch.tensor(0.0, device=self.device)
        
        # 메모리 정리
        torch.cuda.empty_cache()

        sampled = self.sample()

        # 배치 크기를 줄여서 메모리 사용량 감소
        mini_batch_size = min(2, len(sampled))  # 한 번에 최대 2개씩만 처리
        
        for batch_start in range(0, len(sampled), mini_batch_size):
            batch_end = min(batch_start + mini_batch_size, len(sampled))
            mini_batch = sampled[batch_start:batch_end]

            batch_loss = torch.tensor(0.0, requires_grad=True, device=self.device)

            self.rl_updated_model.to(self.device)

            for element in mini_batch:
                seq = element['generated_tokens'].to(self.device)
                question = element['question'].to(self.device)
                step_rewards = torch.tensor([element[0] for element in element['step_rewards']]).to(self.device)
                mean_reward = element['mean_reward']
                
                # For ESM2, we need to calculate probabilities for each position
                sequence_str = self.tokenizer.decode(seq, skip_special_tokens=True)
                
                # Calculate forward probabilities
                forward_probs = []

                # 시퀀스 길이 제한으로 메모리 사용량 감소
                # max_seq_length = min(len(sequence_str), 100)  # 최대 100개 아미노산까지만 처리
                max_seq_length = len(sequence_str)
                print(f"Processing sequence of length {len(sequence_str)} with max_seq_length {max_seq_length}")
                sequence_str = sequence_str[:max_seq_length]
                
                # 메모리 효율적인 처리를 위해 위치별로 하나씩 처리
                for i in range(min(len(seq[:-1]), max_seq_length)):
                    if i < len(sequence_str):
                        try:
                            masked_seq = list(sequence_str)
                            masked_seq[i] = self.tokenizer.mask_token
                            masked_seq_str = ''.join(masked_seq)
                            
                            inputs = self.tokenizer(masked_seq_str, return_tensors="pt").to(self.device)
                            
                            # Gradient checkpointing을 사용하여 메모리 절약
                            with torch.cuda.amp.autocast():  # Mixed precision 사용
                                outputs = self.rl_updated_model(**inputs)
                                logits = outputs.logits[0, i + 1, 4:24]  # ESM2 amino acid range
                                log_probs = F.log_softmax(logits, dim=-1)
                                
                                # Get probability for the actual token
                                actual_token_id = seq[i].item() - 4  # Adjust for ESM2 token offset
                                if 0 <= actual_token_id < 20:
                                    forward_probs.append(log_probs[actual_token_id])
                            
                            # 즉시 메모리 정리
                            del inputs, outputs, logits, log_probs
                            torch.cuda.empty_cache()
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                # print(f"OOM at position {i}, skipping...")
                                # 메모리 부족 시 현재 위치 건너뛰기
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e
                
                if len(forward_probs) > 0:
                    forward_probs = torch.stack(forward_probs)
                    
                    # Calculate GFlowNet loss (simplified version)
                    # This is a basic implementation - you may need to adjust based on your specific GFlowNet formulation
                    rewards_tensor = step_rewards[:len(forward_probs)]
                    if len(rewards_tensor) > 0:
                        gfn_loss = torch.mean((forward_probs - torch.log(rewards_tensor + 1e-8)) ** 2)
                        batch_loss = batch_loss + gfn_loss
                
                sum_rewards += torch.mean(step_rewards)
                sum_mean_reward += mean_reward

                # 메모리 정리
                del seq, question, step_rewards
                torch.cuda.empty_cache()
            
            loss = loss + batch_loss

            # 각 미니배치 후 모델을 CPU로 이동하여 메모리 절약
            self.rl_updated_model.to('cpu')
            torch.cuda.empty_cache()

        avg_rewards = sum_rewards / self.batch_size if self.batch_size > 0 else torch.tensor(0.0)
        avg_mean_reward = sum_mean_reward / self.batch_size if self.batch_size > 0 else torch.tensor(0.0)
        loss = loss / self.batch_size if self.batch_size > 0 else loss
        
        return loss, avg_rewards, avg_mean_reward, sampled

    def mask_sequence(self, sequence, mask_pos):
        """Mask a single position in the sequence and return the masked sequence."""
        masked_sequence = list(sequence)
        masked_sequence[mask_pos] = self.tokenizer.mask_token
        return ''.join(masked_sequence)

    def hamming_distance(self, s1, s2):
        """Calculates the Hamming distance between two sequences"""
        return sum(1 for x, y in zip(s1, s2) if x != y and x != '-' and y != '-')
    
    def hamming_distance_tensor(self, t1, t2):
        """Calculate the Hamming distance between two tensors."""
        return torch.sum(t1 != t2)
    
    def average_pairwise_hamming_distance(self, mutated_seqs):
        """Calculate the average pairwise Hamming distance of a batch of protein sequences for all pairs."""
        batch_size = len(mutated_seqs)
        protein_tensors = torch.zeros((batch_size, len(self.WT)), dtype=torch.bfloat16).to(self.device)
        for i, seq in enumerate(mutated_seqs):
            protein_tensors[i] = torch.tensor(self.aa2ind(list(seq))).to(self.device)
            # print('protein_tensors', protein_tensors[i])

        n = protein_tensors.size(0)
        total_distance = 0
        num_pairs = 0
    
        # Iterate over all unique pairs
        for i, j in itertools.combinations(range(n), 2):
            total_distance += self.hamming_distance_tensor(protein_tensors[i], protein_tensors[j])
            num_pairs += 1
        average_distance = total_distance / num_pairs # Calculate average distance
        # print('average_distance', average_distance)

        ###### Does not save any memory to delete protein_tensors ######
        
        return average_distance, total_distance, num_pairs

    def identify_high_conf_mutations(self, log_probs, tokenizer, WT, high_conf_threshold):
        """Identify high-confidence mutations based on probabilities exceeding the threshold."""
        all_tokens = list(tokenizer.get_vocab().keys())[4:24]  # ESM2 amino acid tokens
        WT_token_ids = [tokenizer.convert_tokens_to_ids(wt) - 4 for wt in WT if wt in tokenizer.get_vocab()]
        
        high_conf_mutations = {}
        for pos, wt_token_id in enumerate(WT_token_ids):
            if pos < log_probs.shape[0]:
                pos_probs = torch.exp(log_probs[pos]).to(self.device)
                high_conf_tokens = [
                    (all_tokens[token_id], prob.item())
                    for token_id, prob in enumerate(pos_probs)
                    if token_id != wt_token_id and prob > high_conf_threshold
                ]
                if high_conf_tokens:
                    high_conf_mutations[pos + 1] = high_conf_tokens
                    
        return high_conf_mutations


    def generate_mutated_sequence(self, wt_sequence, generation_id):
        """Generate a mutated sequence using ESM2 model"""
        print("Generating mutated sequence...")
        # Get the sequence string
        sequence = wt_sequence if isinstance(wt_sequence, str) else wt_sequence[0]
        print(f"WT Sequence: {sequence}")


        ##--------log probabilities--------#
        # Calculate log probabilities for each position
        log_states = torch.zeros((len(sequence), 20), dtype=torch.bfloat16).to(self.device)
        
        with torch.no_grad():
            self.rl_updated_model.to(self.device)
            self.rl_updated_model.eval()
            for mask_pos in range(len(sequence)):
                masked_sequence = self.mask_sequence(sequence, mask_pos)
                inputs = self.tokenizer(masked_sequence, return_tensors="pt").to(self.device)
                logits = self.rl_updated_model(**inputs).logits[:,:,4:24]
                log_probabilities = F.log_softmax(logits[0, mask_pos + 1], dim=-1)
                log_states[mask_pos] = log_probabilities
            
            self.rl_updated_model.train()
        
        ##--------action--------#
        # 1. 동적 온도값 조정 (시간과 generation_id에 따라)
        dynamic_temp = self.temperature * (1 + 0.2 * np.sin(generation_id * np.pi / 4))
        dynamic_temp = max(0.3, min(1.2, dynamic_temp))
        
        # 2. 동적 임계값 조정
        dynamic_threshold = self.high_conf_threshold + 0.1 * np.cos(generation_id * np.pi / 3)
        dynamic_threshold = max(0.4, min(0.8, dynamic_threshold))

        # Identify high confidence mutations
        high_conf_mutations = self.identify_high_conf_mutations(log_states, self.tokenizer, sequence, dynamic_threshold)
        print(f"High confidence mutations identified at positions: {list(high_conf_mutations.keys())}")

        # Apply high confidence mutations
        mutated_seq = list(sequence)
        for pos, mutations in high_conf_mutations.items():
            if mutations:
                max_token, max_prob = max(mutations, key=lambda x: x[1])
                if pos - 1 < len(mutated_seq):
                    mutated_seq[pos - 1] = max_token
        
        # Add additional random mutations up to num_mutations
        positions_to_mutate = list(range(len(sequence)))
        random.shuffle(positions_to_mutate)
        
        current_mutations = self.hamming_distance(''.join(mutated_seq), sequence)
        for pos in positions_to_mutate:
            if current_mutations >= self.num_mutations:
                break
                
            if pos < len(mutated_seq) and mutated_seq[pos] == sequence[pos]:
                masked_seq = mutated_seq.copy()
                masked_seq[pos] = self.tokenizer.mask_token
                masked_seq_str = ''.join(masked_seq)
                
                inputs = self.tokenizer(masked_seq_str, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    self.rl_updated_model.eval()
                    logits = self.rl_updated_model(**inputs).logits[0, pos + 1, 4:24]
                    probabilities = F.softmax(logits / self.temperature, dim=-1)
                    
                    sampled_idx = torch.multinomial(probabilities, 1).item()
                    new_amino_acid_id = sampled_idx + 4
                    new_amino_acid = self.tokenizer.convert_ids_to_tokens([new_amino_acid_id])[0]
                    
                    if new_amino_acid != sequence[pos]:
                        mutated_seq[pos] = new_amino_acid
                        current_mutations += 1

        # Clear the GPU memory cache
        if torch.cuda.is_available():
            self.rl_updated_model.to('cpu')
            torch.cuda.empty_cache() # Frees 2.722 GB
            print("Cleared GPU cache after log probability calculation.")

        mutated_seq = ''.join(mutated_seq)
        print(f"Mutated Sequence: {mutated_seq}")

        return mutated_seq

    def generate(self, wt_sequence):
        """Generate mutated sequences and calculate rewards"""
        mean_rewards = torch.tensor([], device=self.device)
        ###print(f"Original sequences: {wt_sequence}")

        generated_sequences = []
    
        # 여러 번 생성하여 다양성 확보
        for i in range(self.number_generation):
            mutated_seq = self.generate_mutated_sequence(wt_sequence[0], generation_id=i)
            generated_sequences.append(mutated_seq)

        # Calculate rewards for generated sequences
        if len(generated_sequences) > 0:
            fitness_values, rel_WT_fitness = self.calculate_reward(generated_sequences)
            
            # Store sequences in replay buffer with rewards
            for i, seq in enumerate(generated_sequences):
                # Create dummy step rewards and other required data
                step_rewards = [(fitness_values.item(), True)]
                mean_reward = fitness_values.item()
                
                # Convert sequence to tokens for storage
                seq_tokens = torch.tensor(self.tokenizer.encode(seq), device=self.device)
                prompt_tokens = torch.tensor(self.tokenizer.encode(self.WT), device=self.device)
                
                self.ReplayBuffer.add(
                    prompt_tokens, seq_tokens, step_rewards, mean_reward, 10
                )

        mean_rewards = fitness_values if len(generated_sequences) > 0 else torch.tensor(0.0)
        return mean_rewards

    def step(self):
        torch.cuda.empty_cache()
        
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        loss, reward, mean_reward, sampled = self.calculate_loss_from_replay()
        print(f"Calculated loss: {loss.item():.4f} ; Average reward: {mean_reward.item():.2f}")

        self.rl_updated_model.to(self.device)
        optimizer.zero_grad()
        self.manual_backward(loss)

        # Gradient clipping
        max_grad_norm = 5.0
        torch.nn.utils.clip_grad_norm_(self.rl_updated_model.parameters(), max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        self.log("loss", loss.item(), prog_bar=True)
        self.log("average_reward", reward.item(), prog_bar=True)
        self.log("learning_rate", scheduler.get_last_lr()[0], prog_bar=True)
        
        print(f"Training loss: {loss.item():.4f} ; Reward: {reward.item():.2f} ; Replay buffer size: {len(self.ReplayBuffer.buffer)}")
        
        return loss.item(), reward.item()
    
    def training_step(self, batch):
        # Generate sequences and calculate rewards
        wt_sequence, wt_sequence_len = batch
        mean_rewards = self.generate(wt_sequence)
        print(f"Mean rewards from generation: {mean_rewards}")

        # Perform GFlowNet training step if replay buffer has enough samples
        if len(self.ReplayBuffer.buffer) >= self.batch_size:
            loss, reward = self.step()
            print(f"Training step completed. Loss: {loss}, Reward: {reward}")
        else:
            print("Not enough samples in replay buffer to perform training step.")
            return torch.tensor(0.0, requires_grad=True)
        
        if (self.current_epoch != 0) & ((self.current_epoch+1) % self.save_every_n_epochs == 0):
            # Use the logger version number in the filename
            self.save_rl_updated_model()
            print(f'Saving models at epoch {self.current_epoch}')

    def save_model(self, path):
        """Save the model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    # def save_rl_updated_model(self):
    #     """
    #     Save the state dictionary of the rl_updated_vae model to a file, for both the non-EMA and EMA-applied versions.
    #     """
    #     self.model.to(self.device)
    #     # Hyperparameters regarding model saving
    #     ema = ExponentialMovingAverage(self.rl_updated_model.parameters(), decay=0.8).to(self.device)
    #     self.filepath = filepath
    #     self.logger_version = logger_version
        
    #     version = self.logger_version if hasattr(self.logger, 'version') else 'unknown_version'
    #     device_name = "cuda" if torch.cuda.is_available() else "cpu"
    #     base_path = f'./logs/{self.filepath}_{self.model_identifier}/version_{version}'
    #     path_to_non_ema_model = f'{base_path}/non_ema_aligned_{self.model_identifier}_v{version}_ep{self.current_epoch}.pt'
    #     path_to_ema_model = f'{base_path}/ema_aligned_{self.model_identifier}_v{version}_ep{self.current_epoch}.pt'

    #     try:
    #         # # Save the non-EMA version of the model
    #         # torch.save(self.rl_updated_model.state_dict(), path_to_non_ema_model)
    #         # print(f"Saved non-EMA {self.model_identifier} model to {path_to_non_ema_model}")

    #         # Save the EMA version of the model
    #         self.ema.store(self.rl_updated_model.parameters())  # Store the original weights of rl_updated_model
    #         self.ema.copy_to(self.rl_updated_model.parameters())  # Apply EMA weights to rl_updated_model
    #         torch.save(self.rl_updated_model.state_dict(), path_to_ema_model)
    #         self.ema.restore(self.rl_updated_model.parameters())  # Restore the original weights after saving
    #         print(f"Saved EMA {self.model_identifier} model to {path_to_ema_model}")

    #     except Exception as e:
    #         print(f"An error occurred while saving the models: {e}")

    #     self.ema.to('cpu')
    #     self.rl_updated_model.to('cpu')

#==============================================

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ProtRepDatasetESM2(torch.utils.data.Dataset):
    def __init__(self, wt_sequence):
        self.wt_sequence = wt_sequence

    def __len__(self):
        return 1 # 1 sequence

    def __getitem__(self, idx):
        # Return the protein sequence as a string and its length
        return self.wt_sequence, len(self.wt_sequence)

class ProtDataModuleESM2(pl.LightningDataModule):
    def __init__(self, WT, batch_size, seed):
        super().__init__()
        self.wt_sequence = WT
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = ProtRepDatasetESM2(self.wt_sequence)

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        return DataLoader(
            self.train_ds,  # The dataset to load, in this case, the training dataset
            batch_size=self.batch_size,  # The number of samples in each batch to load
            shuffle=True,  # Enable shuffling to randomize the order of data before each epoch
            worker_init_fn=seed_worker,  # Function to initialize each worker's seed to ensure reproducibility across runs
            generator=generator,  # Specify the generator used for random number generation in shuffling
        )

class ProtDataModuleESM2_DDP(pl.LightningDataModule):
    def __init__(self, WT, batch_size, seed):
        super().__init__()
        self.wt_sequence = WT
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = ProtRepDatasetESM2(self.wt_sequence)

    def train_dataloader(self):
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        else:
            generator.manual_seed(2549)

        # Detect if CUDA is available, and adjust the sampler accordingly
        if torch.cuda.is_available():
            print('Loading data to GPU')
            sampler = DistributedSampler(self.train_ds, shuffle=True)
            pin_memory = True
        else:
            print('Loading data to CPU')
            sampler = None  # No distributed sampling for CPU
            pin_memory = False

        return DataLoader(
            self.train_ds,  # Dataset to load
            batch_size=self.batch_size,  # Number of samples in each batch
            sampler=sampler,  # DistributedSampler to split data across GPUs, if applicable
            shuffle=not torch.cuda.is_available(),  # Shuffle data on CPU
            num_workers=8,  # Adjust as needed depending on your system's resources
            worker_init_fn=seed_worker,  # Function to seed each worker
            generator=generator,  # Random number generator for shuffling
            pin_memory=pin_memory,  # Improve data loading performance if you're using GPUs
        )