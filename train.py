import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from environment import Environment

class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masks = [] # Store masks!

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.masks.clear()

# 1. Define the Neural Network (Actor-Critic)
class PPOAgent(nn.Module):
    def __init__(self, input_shape, action_space_size):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 128),
            nn.ReLU()
        )

        self.critic = nn.Linear(128, 1)

        self.actor = nn.Linear(128, action_space_size)

    def forward(self, x):
        # Convert state to float tensor and add batch dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        features = self.feature_extractor(x)
        
        # Get the Value from Critic
        value = self.critic(features)
        
        # Get the logits (unnormalized probabilities) for the flattened action from Actor
        logits = self.actor(features)
        
        return logits, value

    def get_action(self, state, action_mask=None):
        logits, value = self.forward(state)
        
        # Apply Action Masking
        if action_mask is not None:
            # Ensure boolean array and match logits shape
            mask = action_mask.bool()
            # Replace invalid action logits (False) with -infinity so prob becomes 0
            logits = logits.masked_fill(~mask, float('-1e9'))
            
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value

    def evaluate(self, state, action, action_mask=None):
        """ Evaluates a batch of states and old actions for the PPO update """
        logits, value = self.forward(state)
        
        # Apply Action Masking during evaluation
        if action_mask is not None:
            mask = action_mask.bool()
            logits = logits.masked_fill(~mask, float('-1e9'))
            
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_probs, value.squeeze(), entropy

# 2. Main Training Loop Setup
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    env = Environment()
    
    # env.action_space.n gives us the total size (12288)
    action_size = env.action_space.n
    obs_shape = env.observation_space.shape
    
    # Initialize our PyTorch model, optimizer, and buffer
    agent = PPOAgent(input_shape=obs_shape, action_space_size=action_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    buffer = PPOBuffer()
    
    # Hyperparameters
    epochs = 500
    max_steps_per_episode = 100
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        
        epoch_reward = 0
        
        # This is where we collect "rollouts" (experience)
        for step in range(max_steps_per_episode):
            
            # Fetch mask exactly as output from your new env method 
            mask = torch.tensor(env.get_valid_mask(), dtype=torch.bool).to(device)
            
            # 1. Ask the neural network for an action
            action, log_prob, value = agent.get_action(state, action_mask=mask)
            
            # 2. Execute the action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
            
            # 3. Store experience in buffer
            buffer.states.append(state)
            buffer.actions.append(action)
            # Log prob is now 1D so we dont sum it.
            buffer.log_probs.append(log_prob)
            buffer.values.append(value)
            buffer.rewards.append(reward)
            buffer.dones.append(terminated or truncated)
            buffer.masks.append(mask)
            
            epoch_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
                
        # ======== 4. PPO UPDATE ========
        # Step A: Calculate Discounted Returns (How good was the final outcome?)
        returns = []
        discounted_sum = 0
        gamma = 0.99 # Discount factor
        
        for reward, is_done in zip(reversed(buffer.rewards), reversed(buffer.dones)):
            if is_done:
                discounted_sum = 0
            discounted_sum = reward + (gamma * discounted_sum)
            # Insert at the beginning so order remains correct
            returns.insert(0, discounted_sum)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Step B: Convert buffer lists to PyTorch Tensors
        old_states = torch.stack(buffer.states).detach().to(device)
        old_actions = torch.tensor(buffer.actions).detach().to(device)
        old_log_probs = torch.stack(buffer.log_probs).detach().to(device)
        old_values = torch.cat(buffer.values).squeeze().detach().to(device)
        old_masks = torch.stack(buffer.masks).detach().to(device)
        
        # Step C: Calculate Advantages (Was the outcome better than the Critic predicted?)
        advantages = returns - old_values
        # Normalize advantages to stabilize training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Step D: Update network K times (PPO uses multiple epochs on the same data)
        K_epochs = 4
        eps_clip = 0.2
        
        for _ in range(K_epochs):
            # Ask the network to evaluate the OLD states and actions again with NEW weights
            log_probs, values, entropy = agent.evaluate(old_states, old_actions, action_mask=old_masks)
            
            # 1. Ratio of New Probabilities vs Old Probabilities
            # Exponentiating logs gives standard probability division: (new / old)
            ratios = torch.exp(log_probs - old_log_probs)
            
            # 2. PPO Clipping Objective (The core of Proximal Policy Optimization)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            
            # Minimizing negative is maximizing positive
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 3. Critic Loss (Mean Squared Error between predicted value and actual return)
            critic_loss = nn.MSELoss()(values, returns)
            
            # 4. Total Loss (Actor + 0.5 * Critic - 0.01 * Exploration Bonus)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            
            # Backpropagation update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Clear buffer for the next rollout
        buffer.clear()
                
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Total Reward: {epoch_reward:.2f}")

if __name__ == "__main__":
    train()