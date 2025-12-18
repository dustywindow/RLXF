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
                 sft_model,
                 reward_model,
                 accelerator,
                 WT,
                 predicted_WT_fitness,
                 save_every_n_epochs,
                 pairwise_hd_aver_factor=1.0e-06,
                 learning_rate=3e-6,
                 number_generation=2,
                 subTB_lambda=1.0,
                 temperature=0.6,
                 max_new_tokens=800,
                 num_mutations=5, #15 # number of mutations to add to WT
                 high_conf_threshold=0.7, #0.9 # initial probability threshold to be considered high confidence mutation
                 cum_prob_threshold=0.1, #0.22164310879955906,  # initial cumulative probability threshold of non-WT resides to be considered candidate position to explore mutating
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
        self.fixed_model = sft_model
        self.fixed_model.bfloat16()
        self.fixed_model.to('cpu') # Do not need to clear cache. 0 MB freed
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
        self.fixed_high_conf_seq = None
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
    
    def calculate_reward(self, mutated_seqs, pretrained_mutated_seqs):
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
                #     score = reward_model.predict(seq)[0][0]  # Extract score for the sequence from the current model
                #     print(f"Score for sequence {j} by model {i}: {score}")
                #     pre_scores_tensor[i, j] = score

        # Unload all reward models from the GPU
        for reward_model in self.reward_models:
            reward_model.to('cpu')
        
        # Calculate fitness
        predicted_WT_fitness = self.predicted_WT_fitness # Predicted WT score
        rl_fitness_per_sequence = torch.quantile(scores_tensor, 0.05, dim=0)
        pre_fitness_per_sequence = torch.quantile(pre_scores_tensor, 0.05, dim=0)
        print(f"RL-updated mean fitness: {rl_fitness_per_sequence.mean()}")
        print(f"Pre-trained mean fitness: {pre_fitness_per_sequence.mean()}")

        # Compute the overall fitness score based on average_type
        rl_fitness = rl_fitness_per_sequence.max()
        pre_fitness = pre_fitness_per_sequence.max()
        rel_WT_fitness = rl_fitness / predicted_WT_fitness

        if self.rel_to_WT == 1:
            fitness_advantage = rel_WT_fitness
        else:
            fitness_advantage = ((rl_fitness - pre_fitness)/pre_fitness)*100
            # fitness_advantage = rl_fitness

        self.current_rel_WT_fitness = rel_WT_fitness.item()

        pairwise_hd_aver, total_distance, num_pairs = self.average_pairwise_hamming_distance(mutated_seqs)
        # total_reward = (fitness_advantage + self.pairwise_hd_aver_factor*pairwise_hd_aver - current_beta * dkl_value)
        # total_reward = (fitness_advantage + self.pairwise_hd_aver_factor*pairwise_hd_aver)
        total_reward = fitness_advantage
        # total_reward = rl_fitness

        # Logging
        self.log("pairwise_hd_aver", pairwise_hd_aver, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("fitness_advantage", fitness_advantage, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("rl_fitness", rl_fitness, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("rel_WT_fitness", rel_WT_fitness, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("total_reward", total_reward, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log('num_muts', float(self.num_mutations), on_step=True, on_epoch=False, prog_bar=False, logger=True)

        # return fitness_advantage, rel_WT_fitness
        return total_reward, rel_WT_fitness

    def sample(self):
        return self.ReplayBuffer.sample_weighted_and_remove(self.batch_size)
        #return self.ReplayBuffer.sample_weighted(self.batch_size)

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
                # print(f"Processing sequence of length {len(sequence_str)} with max_seq_length {max_seq_length}")
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

                # if len(forward_probs) > 0:
                #     # Calculate GFlowNet loss (simplified version)
                #     log_pf = forward_probs.sum()
                #     final_reward = step_rewards[0]
                #     gfn_loss = torch.mean((log_pf - torch.log(final_reward + 1e-8)) ** 2)
                #     batch_loss = batch_loss + gfn_loss
                
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
    
    def log_probabilities(self, model, sequence=None):
        # Calculate log probabilities for each position
        log_states = torch.zeros((len(sequence), 20), dtype=torch.bfloat16).to(self.device)
        
        if sequence is None:
            sequence = self.WT
        
        with torch.no_grad():
            model.to(self.device)
            model.eval()
            for mask_pos in range(len(sequence)):
                masked_sequence = self.mask_sequence(sequence, mask_pos)
                inputs = self.tokenizer(masked_sequence, return_tensors="pt").to(self.device)
                logits = model(**inputs).logits[:,:,4:24]
                log_probabilities = F.log_softmax(logits[0, mask_pos + 1], dim=-1)
                log_states[mask_pos] = log_probabilities

            model.train()

        return log_states

    def identify_high_conf_mutations(self, log_probs, tokenizer, WT, high_conf_threshold, num_muts):
        """Identify high-confidence mutations with adaptive threshold and mutation count adjustment."""
        max_high_conf_threshold = 0.99
        
        while True:
            all_tokens = list(tokenizer.get_vocab().keys())[4:24]  # ESM2 amino acid tokens
            WT_token_ids = [tokenizer.convert_tokens_to_ids(wt) - 4 for wt in WT if wt in tokenizer.get_vocab()]
            
            high_conf_mutations = {}
            for pos, wt_token_id in enumerate(WT_token_ids):
                pos_probs = torch.exp(log_probs[pos]).to(self.device)
                high_conf_tokens = [
                    (all_tokens[token_id], prob.item())
                    for token_id, prob in enumerate(pos_probs)
                    if token_id != wt_token_id and prob > high_conf_threshold
                ]
                if high_conf_tokens:
                    high_conf_mutations[pos + 1] = high_conf_tokens
            
            # print(f"Found {len(high_conf_mutations)} high-confidence positions with threshold {high_conf_threshold:.3f}")
            
            # 만약 high-confidence mutation 개수가 num_muts보다 적으면 종료
            if len(high_conf_mutations) < num_muts:
                print(f"High-confidence mutations ({len(high_conf_mutations)}) < target mutations ({num_muts}). Accepting current mutations.")
                break
                
            # threshold가 최대값에 도달하면 num_muts 증가
            if high_conf_threshold >= max_high_conf_threshold:
                num_muts = len(high_conf_mutations) + 1  # num_muts 자동 증가!
                print(f"Max threshold reached. Increasing num_muts from {num_muts-1} to {num_muts}.")
                break
                
            # threshold 증가하여 high-confidence mutation 개수 줄이기 시도
            high_conf_threshold = min(high_conf_threshold * 1.01, max_high_conf_threshold)
            # print(f"Too many high-confidence mutations ({len(high_conf_mutations)}). Increasing threshold to {high_conf_threshold:.3f}")
        
        print(f"Final: {len(high_conf_mutations)} high-confidence mutations, num_muts={num_muts}, threshold={high_conf_threshold:.3f}")
        return high_conf_mutations, num_muts, high_conf_threshold

    def identify_candidate_positions(self, log_states, WT, cum_prob_threshold, tokenizer, for_aligned_model=False):
        """
        Identify candidate positions with cumulative probability > threshold for non-wildtype amino acids.
        Args:
            new_log_states (torch.Tensor): Log probabilities for each position (shape: num_positions x vocab_size).
            WT : Wild-type string.
            cum_prob_threshold (float): Threshold for cumulative probability to consider a position.
        Returns:
            rl_candidate_positions (list): Indices of candidate positions.
            rl_normalized_weights (list): Normalized weights for candidate positions
        """
        WT_tokens = [tokenizer.convert_tokens_to_ids(wt) - 4 for wt in WT]
        probabilities = torch.exp(log_states).to(self.device)

        # Re-identify candidate positions until there are at least 5 positions
        max_iterations = 100  # 최대 반복 횟수 제한
        iteration = 0
        while iteration < max_iterations:
            rl_candidate_positions = []
            rl_position_weights = []
        
            # Calculate cumulative probability for non-wildtype amino acids
            for i, position_probs in enumerate(probabilities):
                non_wt_prob = position_probs.sum() - position_probs[WT_tokens[i]]
                if non_wt_prob > cum_prob_threshold:
                    rl_candidate_positions.append(i)
                    rl_position_weights.append(non_wt_prob.item())

            if len(rl_candidate_positions) >= 25:  # Stop if the number of candidate positions drops below threshold
                break

            # Decrease threshold by 5% if len(rl_candidate_positions) < 5
            cum_prob_threshold *= 0.99
            print(f"Threshold decreased to {cum_prob_threshold:.4f} due to insufficient candidate positions.")
            iteration += 1

        # 반복 횟수 초과 시 경고 메시지
        if iteration >= max_iterations:
            print(f"Warning: Maximum iterations ({max_iterations}) reached. Using {len(rl_candidate_positions)} candidate positions.")

        rl_total_weight = sum(rl_position_weights)
        rl_normalized_weights = [w / rl_total_weight for w in rl_position_weights] if rl_total_weight > 0 else []

        # Print detailed information about candidate positions
        if for_aligned_model:
            print(f"Number of candidate positions: {len(rl_candidate_positions)}")

        return rl_candidate_positions, rl_normalized_weights, cum_prob_threshold

    def generate_mutated_sequence(self, wt_sequence):
        """ACTION: Generate a mutated sequence using ESM2 model with adaptive mutation count"""
        print("Generating mutated sequence...")
        # Get the sequence string
        wt_sequence = wt_sequence if isinstance(wt_sequence, str) else wt_sequence[0]
        print(f"WT Sequence: {wt_sequence}")

        ##--------log probabilities--------#
        # Store initial values
        initial_num_muts = self.num_mutations

        if self.current_epoch == 0:
            self.fixed_model.to(self.device)
            # Generate single mutant log probs for fixed model during the first epoch
            self.init_log_states = self.log_probabilities(self.fixed_model, sequence=wt_sequence)
            self.fixed_model.to('cpu')

        self.rl_updated_model.to(self.device)
        new_log_states = self.log_probabilities(self.rl_updated_model, sequence=wt_sequence)
        self.rl_updated_model.to('cpu')

        ##--------identify_high_conf_mutations--------#
        while True:    
            # # 1. 동적 온도값 조정 (시간과 generation_id에 따라)
            # dynamic_temp = self.temperature * (1 + 0.2 * np.sin(generation_id * np.pi / 4))
            # dynamic_temp = max(0.3, min(1.2, dynamic_temp))
            
            # # 2. 동적 임계값 조정
            # dynamic_threshold = self.high_conf_threshold + 0.1 * np.cos(generation_id * np.pi / 3)
            # dynamic_threshold = max(0.4, min(0.8, dynamic_threshold))

            # Identify high confidence mutations with adaptive adjustment
            rl_high_conf_mutations, self.num_mutations, final_threshold = self.identify_high_conf_mutations(
                new_log_states, self.tokenizer, wt_sequence, self.high_conf_threshold, self.num_mutations
            )
            fixed_high_conf_mutations, self.num_mutations, final_threshold = self.identify_high_conf_mutations(
                self.init_log_states, self.tokenizer, wt_sequence, self.high_conf_threshold, self.num_mutations
            )
            
            print(f"High confidence mutations identified at positions: {list(rl_high_conf_mutations.keys())}")
            
            # num_mutations가 증가했다면 다시 시작
            if self.num_mutations > initial_num_muts:
                print(f"num_mutations increased from {initial_num_muts} to {self.num_mutations}. Restarting action...")
                initial_num_muts = self.num_mutations
                continue  # 루프 재시작
            
            break  # num_mutations가 안정적이면 종료

        ##--------generate_mutated_sequences--------#
        fixed_mutated_seqs = [] # Mutated sequences from fixed model
        rl_mutated_seqs = [] # Mutated sequences from aligned model

        # Calculate single mutant probability space for sequence with high confidence mutations from fixed model (constant throughout training)
        if self.current_epoch == 0:
            # Generate sequences for fixed model for 1st iteration of epoch
            fixed_mutated_seq = list(self.WT)
            for pos, mutations in fixed_high_conf_mutations.items():
                max_token, max_prob = max(mutations, key=lambda x: x[1])
                fixed_mutated_seq[pos - 1] = max_token
            self.fixed_high_conf_seq = "".join(fixed_mutated_seq)
            fixed_sequences_with_high_confidence_mutations = [self.fixed_high_conf_seq] * self.number_generation
            print(f"Generated sequence with high confidence mutations from fixed model: {fixed_high_conf_mutations}")

            self.init_log_states = self.log_probabilities(self.fixed_model, sequence=self.fixed_high_conf_seq)
            self.fixed_candidate_positions, self.fixed_normalized_weights, self.cum_prob_threshold = self.identify_candidate_positions(self.init_log_states, self.WT, self.cum_prob_threshold, self.tokenizer)
            # print('Generated candidate positions from fixed model and normalized weights')
        else:
            fixed_sequences_with_high_confidence_mutations = [self.fixed_high_conf_seq] * self.number_generation

        # Apply high confidence mutations
        self.rl_updated_model.to(self.device)
        rl_mutated_seq = list(wt_sequence)
        positions_to_mask = list(rl_high_conf_mutations.keys())
        for pos, mutations in rl_high_conf_mutations.items():
            if mutations:
                max_token, max_prob = max(mutations, key=lambda x: x[1])
                if pos - 1 < len(rl_mutated_seq):
                    rl_mutated_seq[pos - 1] = max_token
        rl_high_conf_seq = "".join(rl_mutated_seq)
        rl_sequences_with_high_confidence_mutations = [rl_high_conf_seq] * self.number_generation
        print(f"Generated sequence with high confidence mutations from aligned model: {rl_high_conf_seq}")

        # # Create masked sequences by masking the high-confidence mutation positions
        # rl_mutated_seq = list(wt_sequence)
        # for pos in positions_to_mask:
        #     rl_mutated_seq[pos - 1] = self.tokenizer.mask_token  # Adjust for 0-indexed list
        # masked_rl_mutated_seq = "".join(rl_mutated_seq)
        self.rl_updated_model.to(self.device)
        new_log_states_with_high_conf_mutations = self.log_probabilities(self.rl_updated_model, sequence=wt_sequence)
        self.rl_updated_model.to('cpu')

        rl_candidate_positions, rl_normalized_weights, self.cum_prob_threshold = self.identify_candidate_positions(new_log_states_with_high_conf_mutations, self.WT, self.cum_prob_threshold, self.tokenizer, for_aligned_model=True)

        # Add additional random mutations up to num_mutations
        for seq in rl_sequences_with_high_confidence_mutations:
            mutated_seq = list(seq)
            while self.hamming_distance(mutated_seq, self.WT) < self.num_mutations:
                
                # Randomly choose a candidate position
                selected_pos = random.choices(rl_candidate_positions, weights=rl_normalized_weights, k=1)[0]
                # print(f"Selected position {selected_pos} for mutation in sequence {seq_idx}")
                
                # Calculate log prob for amino acid mutation for aligned model
                mutated_seq[selected_pos] = self.tokenizer.mask_token  # Use <mask> token
                masked_seq_str = ''.join(mutated_seq)
                # print('masked_seq_str', masked_seq_str)
                inputs = self.tokenizer(masked_seq_str, return_tensors="pt").to(self.device)
                self.rl_updated_model.to(self.device)
                self.rl_updated_model.eval()
                rl_outputs = self.rl_updated_model(**inputs)
                self.rl_updated_model.train()
                rl_logits = rl_outputs.logits[0, selected_pos + 1, 4:24]
                rl_log_probabilities_pos = F.log_softmax(rl_logits, dim=-1)
                rl_probabilities_pos = torch.exp(rl_log_probabilities_pos).to(self.device)
                # print('fixed_probabilities_pos', fixed_probabilities_pos)
                sampled_idx = torch.multinomial(rl_probabilities_pos, 1).item()
                new_amino_acid_id = sampled_idx + 4 # Map to actual token ID range for amino acids
                new_amino_acid = self.tokenizer.convert_ids_to_tokens([new_amino_acid_id])[0]
                mutated_seq[selected_pos] = new_amino_acid

            # Convert tokenized mutated sequence back to amino acid string
            mutated_seq = ''.join(mutated_seq)
            rl_mutated_seqs.append(mutated_seq)
            
        # Convert tokenized mutated sequence back to amino acid string
        rl_mutated_seq = ''.join(rl_mutated_seq)
        
        # Clear the GPU memory cache
        if torch.cuda.is_available():
            self.rl_updated_model.to('cpu')
            torch.cuda.empty_cache() # Frees 2.722 GB
            print("Cleared GPU cache after log probability calculation.")
        
        # Generate designs with 5 mutations from fixed model
        self.fixed_model.to(self.device)
        mut_idx = self.hamming_distance(self.fixed_high_conf_seq, wt_sequence)
        for seq in fixed_sequences_with_high_confidence_mutations:
            mutated_seq = list(seq)
            # print('mutated_seq', mutated_seq)

            with torch.no_grad(): 
                while self.hamming_distance(mutated_seq, wt_sequence) < self.num_mutations:
                    # Randomly choose a candidate position
                    selected_pos = random.choices(self.fixed_candidate_positions, weights=self.fixed_normalized_weights, k=1)[0]
                    # print('selected_pos', selected_pos)
                    
                    # Calculate log prob for amino acid mutation for aligned model (if site is actually mutated)
                    mutated_seq[selected_pos] = self.tokenizer.mask_token  # Use <mask> token
                    masked_seq_str = ''.join(mutated_seq)
                    # print('masked_seq_str', masked_seq_str)
                    inputs = self.tokenizer(masked_seq_str, return_tensors="pt").to(self.device)
                    self.fixed_model.eval()
                    fixed_outputs = self.fixed_model(**inputs)
                    fixed_logits = fixed_outputs.logits[0, selected_pos + 1, 4:24]  # Adjust this range based on valid amino acid tokens
                    fixed_log_probabilities_pos = F.log_softmax(fixed_logits, dim=-1)
                    fixed_probabilities_pos = torch.exp(fixed_log_probabilities_pos).to(self.device)
                    # print('fixed_probabilities_pos', fixed_probabilities_pos)
                    sampled_idx = torch.multinomial(fixed_probabilities_pos, 1).item()
                    new_amino_acid_id = sampled_idx + 4  # Map to actual token ID range for amino acids
                    new_amino_acid = self.tokenizer.convert_ids_to_tokens([new_amino_acid_id])[0]
                    mutated_seq[selected_pos] = new_amino_acid
                    mut_idx = self.hamming_distance(mutated_seq, wt_sequence)

                # Convert tokenized mutated sequence back to amino acid string
                mutated_seq = ''.join(mutated_seq)
                fixed_mutated_seqs.append(mutated_seq)
    
        # Clear the GPU memory cache
        if torch.cuda.is_available():
            self.fixed_model.to('cpu')
            torch.cuda.empty_cache() # Frees 2.722 GB
            print("Cleared GPU cache after log probability calculation.")

        print(f"Mutated Sequences num: {len(rl_mutated_seqs)}")
        print(f"Fixed Mutated Sequences num: {len(fixed_mutated_seqs)}")

        return rl_mutated_seqs, fixed_mutated_seqs

    def generate(self, wt_sequence):
        """Generate mutated sequences and calculate rewards"""
        mean_rewards = torch.tensor([], device=self.device)
        ###print(f"Original sequences: {wt_sequence}")

        mutated_seqs, pretrained_mutated_seqs = self.generate_mutated_sequence(wt_sequence[0])

        # Calculate rewards for generated sequences
        if len(mutated_seqs) > 0:
            fitness_values, rel_WT_fitness = self.calculate_reward(mutated_seqs, pretrained_mutated_seqs)
            # Store sequences in replay buffer with rewards
            for i, seq in enumerate(mutated_seqs):
                # Create dummy step rewards and other required data
                step_rewards = [(fitness_values.item(), True)]
                mean_reward = fitness_values.item()
                
                # Convert sequence to tokens for storage
                seq_tokens = torch.tensor(self.tokenizer.encode(seq), device=self.device)
                prompt_tokens = torch.tensor(self.tokenizer.encode(self.WT), device=self.device)
                
                self.ReplayBuffer.add(
                    prompt_tokens, seq_tokens, step_rewards, mean_reward, 10
                )

        mean_rewards = fitness_values if len(mutated_seqs) > 0 else torch.tensor(0.0)
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
        
        # if (self.current_epoch != 0) & ((self.current_epoch+1) % self.save_every_n_epochs == 0):
        #     # Use the logger version number in the filename
        #     self.save_rl_updated_model()
        #     print(f'Saving models at epoch {self.current_epoch}')

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