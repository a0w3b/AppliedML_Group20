"""UNI-Vaasa, Applied ML / Group20 Project work: RL-Mario — Evaluation script."""

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import torch
import cv2
import time
import numpy as np
import os
import glob
from collections import deque

from agent import DQN
from utils import preprocess, create_video_writer, MarioViewer

# --- Config ---
MODEL_PATH = "checkpoints/dqn_mario.pt"
VIDEO_PATH = "assets/mario_eval.mp4"
NUM_EVAL_EPISODES = 10
MAX_STEPS = 5000
FRAME_STACK_SIZE = 4

# --- Display Settings ---
SHOW_VISUAL = True  # Set to True to show the game window
DISPLAY_SCALE = 3   # Scale factor for the display (3x larger)
DISPLAY_FPS = 30    # Frames per second for display

# --- Create directories if they don't exist ---
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# --- Utility: Load Latest Best Model ---
def get_latest_model(path_pattern):
    model_files = glob.glob(path_pattern)
    if not model_files:
        return None
    return max(model_files, key=os.path.getctime)

# --- Environment ---
env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, RIGHT_ONLY)
n_actions = env.action_space.n

# --- Agent ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(n_actions).to(device)

# --- Load Model ---
latest_best_model = get_latest_model("checkpoints/best_model_*.pt")
if latest_best_model:
    print(f"📦 Loading model from: {latest_best_model}")
    policy_net.load_state_dict(torch.load(latest_best_model, map_location=device))
elif os.path.exists(MODEL_PATH):
    print(f"📦 Loading model from: {MODEL_PATH}")
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print(f"⚠️  No model found at {MODEL_PATH} or in checkpoints/")
    print("   Please train a model first using: python train.py")
    exit(1)

policy_net.eval()
print(f"✅ Model loaded successfully on {device}")

# --- Visual Display ---
viewer = None
if SHOW_VISUAL:
    viewer = MarioViewer(scale=DISPLAY_SCALE, caption="Mario RL Evaluation")
    print(f"👁️  Visual display enabled (scale: {DISPLAY_SCALE}x)")
else:
    print("👁️  Visual display disabled (set SHOW_VISUAL=True to enable)")

# --- Evaluation ---
total_rewards = []
x_positions = []

for ep in range(NUM_EVAL_EPISODES):
    obs, info = env.reset()
    state = preprocess(obs)
    state_stack = deque([state] * FRAME_STACK_SIZE, maxlen=FRAME_STACK_SIZE)
    state = np.array(state_stack)
    total_reward = 0
    prev_x = info.get('x_pos', 40)

    if ep == 0:
        try:
            video = create_video_writer(VIDEO_PATH)
            video_enabled = True
        except Exception as e:
            print(f"⚠️  Video recording disabled: {e}")
            video = None
            video_enabled = False

    for step in range(MAX_STEPS):
        # Render frame (required for environment)
        frame = env.render()
        
        # Display frame visually
        if viewer is not None:
            viewer.display_frame(frame, fps=DISPLAY_FPS)
        
        # Write to video if enabled
        if ep == 0 and video_enabled and video is not None:
            try:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception:
                video_enabled = False

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = preprocess(next_obs)
        state_stack.append(next_state)
        state = np.array(state_stack)
        total_reward += reward
        prev_x = info.get('x_pos', prev_x)

        if terminated or truncated:
            print(f"🛑 Episode {ep} ended at step {step} — x_pos: {prev_x}")
            break

    total_rewards.append(total_reward)
    x_positions.append(prev_x)

    if ep == 0 and video_enabled and video is not None:
        video.release()
        print(f"📹 Video saved to {VIDEO_PATH}")

if viewer is not None:
    viewer.close()
    print("👁️  Visual display closed")
env.close()

# --- Summary ---
print(f"\n✅ Evaluation Summary over {NUM_EVAL_EPISODES} episodes:")
print(f"🔍 Average Reward: {np.mean(total_rewards):.2f}")
print(f"📍 Average x_pos: {np.mean(x_positions):.2f}")
print(f"🎥 First episode saved to: {VIDEO_PATH}")
