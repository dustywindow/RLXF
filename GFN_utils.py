import torch
import heapq
import numpy as np

def get_newline_token_id(tokenizer):
    all_tokens = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    newline_token_ids = [i for i, token in enumerate(all_tokens) if "\n" in token]
    return newline_token_ids

class ReplayBuffer:
    def __init__(self, max_size):
        """
        Args:
            max_size: Maximum number of trajectories the buffer can hold.
        """
        self.max_size = max_size
        self.buffer = []
        self.min_heap = []

    def remove_high_count(self, max_count=50): #useful for step by step gfn
        """
        Remove all trajectories from the replay buffer with 'count' greater than max_count.
        Args:
            max_count: Maximum allowed count value. Entries with count > max_count will be removed.
        """
        valid_indices = set(idx for idx, entry in enumerate(self.buffer) if entry["count"] <= max_count)

        self.buffer = [entry for entry in self.buffer if entry["count"] <= max_count]

        self.min_heap = [(mean_reward, idx) for mean_reward, idx in self.min_heap if idx in valid_indices]

    def add(self, question, generated_tokens, step_rewards, mean_reward, count):
        """
        Add trajectories to the replay buffer. If max size is reached, only keep the
        trajectories with the highest rewards.
        Args:
            generated_tokens: The generated sequence.
            step_rewards: List of rewards, one for each step.
            mean_reward: Mean reward, used for priority when buffer is full.
        """
        if len(self.buffer) < self.max_size:
            self.buffer.append({
                "question": question,
                "generated_tokens": generated_tokens.cpu(),
                "step_rewards": step_rewards,
                "mean_reward": mean_reward,
                "count": count,
            })
            heapq.heappush(self.min_heap, (mean_reward, len(self.buffer) - 1))
        else:
            if mean_reward > self.min_heap[0][0]:
                _, index_to_remove = heapq.heappop(self.min_heap)
                self.buffer[index_to_remove] = {
                    "question": question,
                    "generated_tokens": generated_tokens,
                    "step_rewards": step_rewards,
                    "mean_reward": mean_reward,
                    "count": count,
                }
                heapq.heappush(self.min_heap, (mean_reward, index_to_remove))
                
                
    def sample_and_remove(self, batch_size):
        """
        Sample trajectories uniformly and remove them from the buffer.

        Args:
            batch_size: Number of trajectories to sample.

        Returns:
            List of sampled trajectories.
        """
        if len(self.buffer) == 0:
            return None

        batch_size = min(batch_size, len(self.buffer))

        # Échantillonnage uniforme
        probabilities = np.ones(len(self.buffer)) / len(self.buffer)

        # Échantillonner des indices dans self.buffer
        sampled_indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probabilities)
        sampled_indices = sorted(sampled_indices, reverse=True)  # Trier pour éviter le décalage des indices lors de la suppression

        # Récupérer les trajectoires échantillonnées
        sampled_trajectories = [self.buffer[i] for i in sampled_indices]

        # Supprimer les trajectoires sélectionnées en partant des indices les plus élevés
        for i in sampled_indices:
            del self.buffer[i]

        return sampled_trajectories

    def sample(self, batch_size):
        """
        Sample trajectories uniformly and keep them in the buffer.

        Args:
            batch_size: Number of trajectories to sample.

        Returns:
            List of sampled trajectories.
        """
        if len(self.buffer) == 0:
            return None

        batch_size = min(batch_size, len(self.buffer))

        # Échantillonnage uniforme
        probabilities = np.ones(len(self.buffer)) / len(self.buffer)

        # Échantillonner des indices dans self.buffer
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probabilities)
        sampled_trajectories = [self.buffer[i] for i in indices]

        return sampled_trajectories
    

    def sample_weighted_and_remove(self, batch_size):
        """
        Sample trajectories proportionally to their mean_reward and remove them from the buffer.

        Args:
            batch_size: Number of trajectories to sample.

        Returns:
            List of sampled trajectories.
        """
        if len(self.buffer) == 0:
            return None

        batch_size = min(batch_size, len(self.buffer))

        mean_rewards = np.array([entry["mean_reward"] for entry in self.buffer])
        total_reward = np.sum(mean_rewards)

        if total_reward == 0:
            probabilities = np.ones(len(self.buffer)) / len(self.buffer)  # Uniform sampling si tous les rewards sont 0.
        else:
            probabilities = mean_rewards / total_reward

        # Échantillonner des indices
        sampled_indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probabilities)
        sampled_indices = sorted(sampled_indices, reverse=True)  # Tri décroissant pour éviter les erreurs de suppression

        # Récupérer les trajectoires échantillonnées
        sampled_trajectories = [self.buffer[i] for i in sampled_indices]

        # Supprimer les trajectoires du buffer
        for i in sampled_indices:
            del self.buffer[i]

        return sampled_trajectories


    def sample_weighted(self, batch_size):
        """
        Sample trajectories proportionally to their mean_reward and keep them in the buffer.

        Args:
            batch_size: Number of trajectories to sample.

        Returns:
            List of sampled trajectories.
        """
        if len(self.buffer) == 0:
            return None

        batch_size = min(batch_size, len(self.buffer))

        mean_rewards = np.array([entry["mean_reward"] for entry in self.buffer])
        total_reward = np.sum(mean_rewards)
        if total_reward == 0:
            probabilities = np.ones(len(self.buffer)) / len(self.buffer)  # Uniform if all rewards are zero.
        else:
            probabilities = mean_rewards / total_reward

        # Échantillonner des indices dans self.buffer
        sampled_indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probabilities)
        sampled_trajectories = [self.buffer[i] for i in sampled_indices]

        return sampled_trajectories

    def reset(self):
        """
        Reset the replay buffer to an empty state.
        """
        self.buffer.clear()
        self.min_heap.clear()
        
        
# class StoppingCriteriaSub(StoppingCriteria):
#     def __init__(self, stops=[], encounters=1, prompt_length=0, initial_ignore=4):
#         """
#         Args:
#             stops: Liste des tokens qui declenchent l'arret.
#             encounters: Nombre de fois qu'un token de `stops` doit etre rencontre pour arreter.
#             prompt_length: Longueur du prompt initial (pour ignorer les tokens de l'entree).
#             initial_ignore: Nombre de tokens a ignorer apres le prompt avant de commencer a verifier.
#         """
#         super().__init__()
#         self.stops = stops
#         self.ENCOUNTERS = encounters
#         self.prompt_length = prompt_length
#         self.initial_ignore = initial_ignore  
#         self.encountered_count = 0 
#         self.processed_tokens = 0  

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        
#         generated_ids = input_ids[0][self.prompt_length:]

#         total_generated = len(generated_ids)

#         if total_generated <= self.initial_ignore:
#             self.processed_tokens = total_generated
#             return False

#         for token_id in generated_ids[self.processed_tokens:]:
#             if any((token_id == stop.to(input_ids.device)).all() for stop in self.stops):
#                 self.encountered_count += 1

#             if self.encountered_count >= self.ENCOUNTERS:
#                 return True

#         self.processed_tokens = total_generated
#         return False