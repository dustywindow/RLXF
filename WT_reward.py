import torch
from torch import cuda

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from MLP import (SeqFcnDataset, ProtDataModule, MLP)

if cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_identifier ='esm2_t33_650M_UR50D'
model_name = f"facebook/{model_identifier}"
num_reward_models = 2
filepath = 'GFlowNets'

# load ensemble of reward models
reward_models = []
for i in range(num_reward_models):
    model_name = f"reward_model_v{i}.ckpt"
    checkpoint_path = f"./reward_models/{model_name}"
    reward_model = MLP.load_from_checkpoint(checkpoint_path)
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_models.append(reward_model)
    
# load ESM2
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_identifier}")
ESM2 = AutoModelForMaskedLM.from_pretrained(f"facebook/{model_identifier}")
ESM2 = ESM2.to(device)

prot_type = 'CreiLOV'  # 'CreiLOV', 'avgfp', 'bgl3'
if prot_type == 'CreiLOV':
    WT = 'MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA'
elif prot_type == 'avgfp':
    WT = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK' # parent sequence
elif prot_type == 'bgl3':
    WT = 'MVPAAQQTAMAPDAALTFPEGFLWGSATASYQIEGAAAEDGRTPSIWDTYARTPGRVRNGDTGDVATDHYHRWREDVALMAELGLGAYRFSLAWPRIQPTGRGPALQKGLDFYRRLADELLAKGIQPVATLYHWDLPQELENAGGWPERATAERFAEYAAIAADALGDRVKTWTTLNEPWCSAFLGYGSGVHAPGRTDPVAALRAAHHLNLGHGLAVQALRDRLPADAQCSVTLNIHHVRPLTDSDADADAVRRIDALANRVFTGPMLQGAYPEDLVKDTAGLTDWSFVRDGDLRLAHQKLDFLGVNYYSPTLVSEADGSGTHNSDGHGRSAHSPWPGADRVAFHQPPGETTAMGWAVDPSGLYELLRRLSSDFPALPLVITENGAAFHDYADPEGNVNDPERIAYVRDHLAAVHRAIKDGSDVRGYFLWSLLDNFEWAHGYSKRFGAVYVDYPTGTRIPKASARWYAEVARTGVLPTAGDPNSSSVDKLAAALEHHHHHH' # parent sequence

print(f"Predicting fitness for WT sequence of length {len(WT)}")

scores_tensor = torch.zeros((len(reward_models), 1), dtype=torch.float32).to(device)
# Compute scores for mutated sequences
with torch.no_grad():
    for i, reward_model in enumerate(reward_models):
        reward_model.eval()  # Set the model to evaluation mode

        score = reward_model.predict(WT)[0][0]  # Extract score for the sequence from the current model
        print(f"Score for sequence by model {i}: {score}")
        scores_tensor[i, 0] = score

# Unload all reward models from the GPU
for reward_model in reward_models:
    reward_model.to('cpu')

predicted_WT_fitness = scores_tensor.mean(dim=0)

print(predicted_WT_fitness)