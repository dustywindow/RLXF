import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from torch_ema import ExponentialMovingAverage
from transformers import AutoModelForMaskedLM, AutoTokenizer
from functions import (generate_df, generate_and_evaluate_mutants_p_sampling)

from MLP import MLP
from GFlowNets import GFlowNet

model_identifier ='esm2_t33_650M_UR50D'
model_name = f"facebook/{model_identifier}"
num_reward_models = 2
filepath = 'GFlowNets'
save_filepath = f'./logs/{filepath}_{model_identifier}'
version = 10
epochs = 1000

# parameters for generating designs after alignment
num_designs = 100
num_muts = 5
high_conf_threshold = 0.9
cum_prob_threshold = 0.25
ep = epochs - 1
generation_seed = 7028

prot_type = 'CreiLOV'  # 'CreiLOV' or 'avgfp'
if prot_type == 'CreiLOV':
    WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA'
    predicted_wt_score = 1.1498 # predicted wildtype score as reference for evaluations
elif prot_type == 'avgfp':
    WT = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK' # parent sequence
   
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")
logger_name = f'SFT_{model_identifier}'

sft_logger_version = 0
sft_model_path = f'./logs/{logger_name}/version_{sft_logger_version}/SFT_{model_identifier}_v{sft_logger_version}.pt'
sft_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")

rl_updated_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
rl_checkpoint_path = f'./logs/{filepath}_{model_identifier}/version_{version}/checkpoints/epoch={ep}-step={epochs}.ckpt'  # 실제 ckpt 파일 경로로 변경

try:
    # Lightning checkpoint 형식인 경우
    checkpoint = torch.load(rl_checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

        # # 모든 키 출력하여 구조 파악
        # i = 0
        # print("All keys in checkpoint:")
        # for key in sorted(state_dict.keys())[:-1]:  # 처음 20개만 출력
        #     # print(f"  {key}")
        #     i += 1
        # print(f"Total keys: {i}")
                
        new_state_dict = {}

        # 먼저 rl_updated_model prefix로 시도
        for key, value in state_dict.items():
            if key.startswith('rl_updated_model.'):
                new_key = key.replace('rl_updated_model.', '')
                new_state_dict[new_key] = value

        # rl_updated_model prefix가 없으면 model prefix로 시도
        if not new_state_dict:
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key.replace('model.', '')
                    new_state_dict[new_key] = value
        
        # 그것도 없으면 전체 state_dict 사용
        if not new_state_dict:
            print("No 'model.' prefix found, using entire state_dict")
            new_state_dict = state_dict
        
        if new_state_dict:
            rl_updated_model.load_state_dict(new_state_dict)
            print(f"Loaded {len(new_state_dict)} parameters")
        else:
            print("No compatible parameters found in checkpoint")
    else:
        # 단순 state_dict 형식인 경우
        rl_updated_model.load_state_dict(checkpoint)
    
    print(f"Successfully loaded RL model from {rl_checkpoint_path}")

except Exception as e:
    print(f"Error loading RL model checkpoint: {e}")
    print("Using pretrained model instead")

# load ensemble of reward models
reward_models = []
num_reward_models = 2
for i in range(num_reward_models):
    model_name = f"reward_model_v{i}.ckpt"
    checkpoint_path = f"./reward_models/{model_name}"
    reward_model = MLP.load_from_checkpoint(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_models.append(reward_model)

###########################################################################################################################################################

"""
Save the state dictionary of the rl_updated_vae model to a file, for both the non-EMA and EMA-applied versions.
"""
# Hyperparameters regarding model saving
ema = ExponentialMovingAverage(rl_updated_model.parameters(), decay=0.8)
ema.to('cuda' if torch.cuda.is_available() else 'cpu')

device_name = "cuda" if torch.cuda.is_available() else "cpu"
base_path = f'./logs/{filepath}_{model_identifier}/version_{version}'
path_to_non_ema_model = f'{base_path}/non_ema_aligned_{model_identifier}_v{version}_ep{ep}.pt'
path_to_ema_model = f'{base_path}/ema_aligned_{model_identifier}_v{version}_ep{ep}.pt'

try:
    # # Save the non-EMA version of the model
    # torch.save(self.rl_updated_model.state_dict(), path_to_non_ema_model)
    # print(f"Saved non-EMA {self.model_identifier} model to {path_to_non_ema_model}")

    # Save the EMA version of the model
    ema.store(rl_updated_model.parameters())  # Store the original weights of rl_updated_model
    ema.copy_to(rl_updated_model.parameters())  # Apply EMA weights to rl_updated_model
    torch.save(rl_updated_model.state_dict(), path_to_ema_model)
    ema.restore(rl_updated_model.parameters())  # Restore the original weights after saving
    print(f"Saved EMA {model_identifier} model to {path_to_ema_model}")

except Exception as e:
    print(f"An error occurred while saving the models: {e}")

ema.to('cpu')
rl_updated_model.to('cpu')

############################################################################################################################################################

# Plot metrics
pt_metrics = pd.read_csv(f'{save_filepath}/version_{version}/metrics.csv')
# Define the metrics you want to plot
metrics_to_plot = [
    ['kl_divergence'],
    ['mean_ratio_initial_iter', 'mean_ratio_final_iter'],
    ['median_ratio_initial_iter', 'median_ratio_final_iter'],
    ['ppo_loss_initial_iter', 'ppo_loss_final_iter'],
    ['fitness_advantage'],
    ['rel_WT_fitness'],
    ['pairwise_hd_aver'],
    ['mean_hd_from_CreiLOV'],
    ['total_reward'],
    ['batch_size'],
    ['num_masks'],
    ['max_norm']]

# Calculate the number of rows for subplots, assuming 1 column
num_rows = len(metrics_to_plot)
# Create subplots
fig, axs = plt.subplots(num_rows, 1, figsize=(10, num_rows * 3))  # Adjust the size as needed
# In case there is only one metric, axs won't be an array, so we make it one for consistency
if num_rows == 1:
    axs = [axs]
# Define ratio metrics for which legends will be added
ratio_metrics = {'mean_ratio_initial_iter', 'mean_ratio_final_iter', 'median_ratio_initial_iter', 'median_ratio_final_iter', 'ppo_loss_initial_iter', 'ppo_loss_final_iter'}
# Loop through each group of metrics and create a plot
for i, metric_group in enumerate(metrics_to_plot):
    for metric in metric_group:
        if metric in pt_metrics.columns:
            data = pt_metrics[~pt_metrics[metric].isna()][metric]
            steps = pt_metrics[~pt_metrics[metric].isna()]['step']
            axs[i].plot(steps, data, label=metric.title())
    
    # Check if the current metric group contains any ratio metrics for adding legends
    if any(metric in ratio_metrics for metric in metric_group):
        axs[i].legend()
    axs[i].set_xlabel('Epoch/Step')
    axs[i].set_ylabel(', '.join(metric_group).replace('_initial_iter', '').replace(', mean_ratio_final_iter', '').replace(', median_ratio_final_iter', '').replace(', ppo_loss_final_iter', '').title())
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
# Adjust the layout and display the plot
fig.tight_layout()
# Save figure
plt.savefig(f'{save_filepath}/version_{version}/metrics_vs_steps.svg')
plt.savefig(f'{save_filepath}/version_{version}/metrics_vs_steps.png')
print('saved learning curves from aligned model')

############################################################################################################################################################

# Load pretrained models
fixed_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")

# Generate and evaluate 1000 designs with 5 mutants
fixed_mutated_seqs, fixed_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, fixed_model, model_identifier, tokenizer, f'{save_filepath}/version_{version}', ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, generation_seed)
print(f"Status: finished generating sequences with fixed {model_identifier}")

# Save mutants from ESM2
base_path = f'{save_filepath}/version_{version}/'
np.save(base_path + f'fixed_{model_identifier}_scores.npy', fixed_scores_np)
with open(base_path + f'fixed_{model_identifier}_mutated_seqs.txt', 'w') as file:
    for seq in fixed_mutated_seqs:
        file.write(seq + '\n')

############################################################################################################################################################

# Load sft model
sft_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
state_dict = torch.load(f'{sft_model_path}')
sft_model.load_state_dict(state_dict)

# Generate and evaluate 1000 designs with 5 mutants from both models
sft_model_identifier = f"SFT_{model_identifier}"
sft_mutated_seqs, sft_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, sft_model, sft_model_identifier, tokenizer, f'{save_filepath}/version_{version}', ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, generation_seed)
print(f"Status: finished generating sequences with sft {model_identifier}")

# Save mutants from ESM2
base_path = f'{save_filepath}/version_{version}/'
np.save(base_path + f'sft_{model_identifier}_scores.npy', sft_scores_np)
with open(base_path + f'sft_{model_identifier}_mutated_seqs.txt', 'w') as file:
    for seq in sft_mutated_seqs:
        file.write(seq + '\n')

############################################################################################################################################################

# Load mutants
fixed_scores_np = np.load(f'{save_filepath}/version_{version}/fixed_{model_identifier}_scores.npy')
fixed_mutated_seqs = []
with open(f'{save_filepath}/version_{version}/fixed_{model_identifier}_mutated_seqs.txt', 'r') as file:
    fixed_mutated_seqs = file.read().splitlines()

sft_scores_np = np.load(f'{save_filepath}/version_{version}/sft_{model_identifier}_scores.npy')
sft_mutated_seqs = []
with open(f'{save_filepath}/version_{version}/sft_{model_identifier}_mutated_seqs.txt', 'r') as file:
    sft_mutated_seqs = file.read().splitlines()

# Generate DataFrames
df_sft = generate_df(sft_mutated_seqs, np.median(sft_scores_np, axis=0), WT)
df_fixed = generate_df(fixed_mutated_seqs, np.median(fixed_scores_np, axis=0), WT)

# Save to CSV
df_sft.to_csv(f'{save_filepath}/version_{version}/{model_identifier}_sft_mutated_designs_scores.csv', index=False)
df_fixed.to_csv(f'{save_filepath}/version_{version}/{model_identifier}_fixed_mutated_designs_scores.csv', index=False)

# Load mutants
fixed_scores_np = np.load(f'{save_filepath}/version_{version}/fixed_{model_identifier}_scores.npy')
fixed_mutated_seqs = []
with open(f'{save_filepath}/version_{version}/fixed_{model_identifier}_mutated_seqs.txt', 'r') as file:
    fixed_mutated_seqs = file.read().splitlines()

sft_scores_np = np.load(f'{save_filepath}/version_{version}/sft_{model_identifier}_scores.npy')
sft_mutated_seqs = []
with open(f'{save_filepath}/version_{version}/sft_{model_identifier}_mutated_seqs.txt', 'r') as file:
    sft_mutated_seqs = file.read().splitlines()

# Load rl models
rl_model = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
state_dict = torch.load(f'{save_filepath}/version_{version}/ema_aligned_{model_identifier}_v{version}_ep{ep}.pt')
rl_model.load_state_dict(state_dict)

# Generate and evaluate 1000 designs with 5 mutants from both models
rl_model_identifier = f"aligned_{model_identifier}"
rl_mutated_seqs, rl_scores_np = generate_and_evaluate_mutants_p_sampling(WT, reward_models, rl_model, rl_model_identifier, tokenizer, f'{save_filepath}/version_{version}', ep, version, num_designs, num_muts, cum_prob_threshold, high_conf_threshold, generation_seed)
print(f"Status: finished generating sequences with sft {model_identifier}")

# Save mutants from ESM2
base_path = f'{save_filepath}/version_{version}/'
np.save(base_path + f'ema_aligned_{model_identifier}_scores.npy', rl_scores_np)
with open(base_path + f'ema_aligned_{model_identifier}_mutated_seqs.txt', 'w') as file:
    for seq in rl_mutated_seqs:
        file.write(seq + '\n')

# Load mutants
rl_scores_np = np.load(f'{save_filepath}/version_{version}/ema_aligned_{model_identifier}_scores.npy')
rl_mutated_seqs = []
with open(f'{save_filepath}/version_{version}/ema_aligned_{model_identifier}_mutated_seqs.txt', 'r') as file:
    rl_mutated_seqs = file.read().splitlines()

# Generate DataFrames
df_rl = generate_df(rl_mutated_seqs, np.median(rl_scores_np, axis=0), WT)

# Save to CSV
df_rl.to_csv(f'{save_filepath}/version_{version}/ema_aligned_{model_identifier}_mutated_designs_scores_ep{ep}.csv', index=False)

# Plot histogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot histograms for the models
sns.histplot(np.median(fixed_scores_np, axis=0), bins=25, alpha=0.4, color='grey', edgecolor='black', stat='density', ax=ax1, label='Pre-trained ESM2')
sns.histplot(np.median(sft_scores_np, axis=0), bins=25, alpha=0.6, color='orange', edgecolor='black', stat='density', ax=ax1, label='SFT ESM2')
sns.histplot(np.median(rl_scores_np, axis=0), bins=25, alpha=0.6, color='blue', edgecolor='black', stat='density', ax=ax1, label='Aligned ESM2')
ax1.set_xlabel('Predicted Fluorescence', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.axvline(predicted_wt_score, color='orange', linestyle='--', linewidth=3)
ax1.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(sft_scores_np, axis=0))-0.05), predicted_wt_score, color='red', alpha=0.1, zorder=-1)
ax1.axvspan(predicted_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(sft_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, zorder=-1)
ax1.legend()

# Plot the cumulative density plot on the second subplot for all models
sns.ecdfplot(np.median(fixed_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="grey", linestyle='-')
sns.ecdfplot(np.median(sft_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="orange", linestyle='-')
sns.ecdfplot(np.median(rl_scores_np, axis=0), stat="proportion", complementary=True, ax=ax2, color="blue", linestyle='-')
ax2.set_xlabel('Predicted Fluorescence', fontsize=12)
ax2.set_ylabel('Cumulative Density', fontsize=12)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axvline(predicted_wt_score, color='orange', linestyle='--', linewidth=3)
ax2.axvspan(min(min(np.median(fixed_scores_np, axis=0))-0.05, min(np.median(sft_scores_np, axis=0))-0.05), predicted_wt_score, color='red', alpha=0.1, zorder=-1)
ax2.axvspan(predicted_wt_score, max(max(np.median(fixed_scores_np, axis=0)) + 0.05, max(np.median(sft_scores_np, axis=0)) + 0.05), color='green', alpha=0.1, label='Better than WT Fluorescence', zorder=-1)
less_wt_patch = mpatches.Patch(color='red', alpha=0.8, label='Less than WT Log Fluorescence')
wt_line = mpatches.Patch(color='orange', alpha=0.8, label='Mean WT Log Fluorescence')
better_wt_patch = mpatches.Patch(color='green', alpha=0.8, label='Greater than WT Log Fluorescence')
legend = ax2.legend(handles=[less_wt_patch, wt_line, better_wt_patch], frameon=True, edgecolor='black')
plt.setp(legend.get_texts(), color='black', fontsize=10)
plt.setp(legend.get_frame(), facecolor='white')
plt.tight_layout()

# Save the plot
plt.savefig(f'{save_filepath}/version_{version}/{model_identifier}_design_scores_ep{ep}.svg')
plt.savefig(f'{save_filepath}/version_{version}/{model_identifier}_design_scores_ep{ep}.png')
print('Saved design histograms')