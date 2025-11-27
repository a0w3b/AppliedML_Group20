"""UNI-Vaasa, Applied ML / Group20 Project work: RL-Mario — training script.
"training_run_0" Pilot training script for DQN agent on Super Mario Bros environment"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym
import torch
import time
import os
import cv2
import glob
from datetime import datetime
from math import ceil

from agent import DQN, ReplayBuffer, select_action
from utils import preprocess, plot_rewards, create_video_writer, MarioViewer

# --- Hyperparameters ---
EPISODES = 100
MAX_STEPS = 5000
GAMMA = 0.99
LR = 0.1 #e-4
BATCH_SIZE = 32
EPS_START = 0.5
EPS_END = 0.05  # Lower epsilon end for more exploitation of learned behavior
EPS_DECAY = 0.995  # Slower decay to explore longer (find high jump strategy)
TARGET_UPDATE = 10
MEMORY_SIZE = 100_000
SAVE_PATH = "checkpoints/dqn_mario.pt"
VIDEO_PATH = "assets/mario_training.mp4"
REWARD_PLOT_PATH = "assets/reward_plot.png"
BEST_REWARD_PATH = "assets/reward.best"

# --- Reward/Training Controls ---
STUCK_THRESHOLD = 150
MAX_STUCK_STEPS = 500
SLOW_PENALTY_DELAY = 15
MAX_SLOW_PENALTY = 5.0
SPRINT_SPEED_LOW = 1.0
SPRINT_SPEED_HIGH = 3.0
GRAD_CLIP_NORM = 5.0
VERBOSE = True

# --- Display Settings ---
SHOW_VISUAL = False  # Set to True to show the game window
DISPLAY_SCALE = 2   # Scale factor for the display (3x larger)
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

# --- Environment Setup ---
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, RIGHT_ONLY)
n_actions = env.action_space.n

# --- Agent Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE, device)


best_reward = float('-inf')

# --- Resume from Checkpoint ---
latest_best_model = get_latest_model("checkpoints/best_model_*.pt")
if latest_best_model:
    print(f"📦 Resuming training from latest best model: {latest_best_model}")
    policy_net.load_state_dict(torch.load(latest_best_model, map_location=device))
    target_net.load_state_dict(policy_net.state_dict())
    try:
        score_file = open(BEST_REWARD_PATH, "r")
        best_reward = float(score_file.read())
        score_file.close()
        print(f"Using the previously saved best reward: {best_reward}")
    except Exception as e:
        print(f"Using default score {best_reward}")
    
elif os.path.exists(SAVE_PATH):
    print(f"📦 Resuming training from {SAVE_PATH}")
    policy_net.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    target_net.load_state_dict(policy_net.state_dict())
    try:
        score_file = open(BEST_REWARD_PATH, "r")
        best_reward = float(score_file.read())
        score_file.close()
        print(f"Using the previously saved best reward: {best_reward}")
    except Exception as e:
        print(f"Using default score {best_reward}")
else:
    print("🚀 Starting fresh training — no checkpoint found")
    print(f"Using the previously saved best reward: {best_reward}")

# --- Video Writer ---
try:
    video = create_video_writer(VIDEO_PATH)
    video_enabled = True
    print(f"📹 Video recording enabled: {VIDEO_PATH}")
except Exception as e:
    print(f"⚠️  Video recording disabled: {e}")
    video = None
    video_enabled = False

# --- Visual Display ---
viewer = None
if SHOW_VISUAL:
    viewer = MarioViewer(scale=DISPLAY_SCALE, caption="Mario RL Training")
    print(f"👁️  Visual display enabled (scale: {DISPLAY_SCALE}x)")
else:
    print("👁️  Visual display disabled (set SHOW_VISUAL=True to enable)")

# --- Training Loop ---
epsilon = EPS_START
reward_history = []
forward_progress_history = []
avg_speed_history = []
air_steps_history = []
high_jump_history = []
max_air_chain_history = []
loss_history = []
epsilon_history = []
stuck_termination_history = []

for episode in range(EPISODES):
    start_time = time.time()
    obs, info = env.reset()
    state = preprocess(obs)
    total_reward = 0
    prev_x = info.get('x_pos', 40)
    prev_score = info.get('score', 0)

    prev_y = info.get('y_pos', 79)  # Track y position for jump detection
    stuck_counter = 0  # Count consecutive steps with no progress
    last_progress_x = prev_x  # Track last position where progress was made
    was_stuck_prev_step = False  # Track if we were stuck in previous step
    
    # Track jump sequences for better rewards
    jump_sequence = []  # Track recent jump actions
    max_y_reached = prev_y  # Track maximum y position (lowest y value = highest jump)
    steps_at_ground = 0  # Count steps at ground level
    last_jump_action = None  # Track last jump action taken
    air_time_steps = 0  # Total steps spent in the air
    current_air_chain = 0  # Consecutive air steps for current jump
    max_air_chain = 0  # Best air streak within episode
    high_jump_count = 0  # Number of high jump actions taken
    episode_forward_progress = 0.0  # Total forward movement
    speed_chain = 0  # Consecutive steps with progress
    max_speed_chain = 0  # Longest streak of forward movement
    no_progress_steps = 0  # Steps without forward movement
    episode_loss_total = 0.0
    loss_updates = 0
    terminated_due_to_stuck = False
    
    for step in range(MAX_STEPS):
        frame = None
        if viewer is not None or (video_enabled and video is not None):
            frame = env.render()
        
        # Display frame visually
        if viewer is not None and frame is not None:
            viewer.display_frame(frame, fps=DISPLAY_FPS)
        
        # Save frame to video
        if video_enabled and video is not None and frame is not None:
            try:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                if step == 0:
                    print(f"⚠️  Video writing failed: {e}")
                    video_enabled = False

        # --- Stuck Detection (before action selection) ---
        # Check if we're currently stuck based on previous step
        is_stuck = stuck_counter > STUCK_THRESHOLD
        
        # Enhanced exploration logic
        approaching_obstacle_before_action = stuck_counter > 50 and stuck_counter <= STUCK_THRESHOLD
        
        # If stuck or approaching obstacle, force more exploration and encourage high jumps
        if is_stuck or approaching_obstacle_before_action:
            # Temporarily increase epsilon to force exploration
            boost = 0.5 if is_stuck else 0.3  # Stronger boost when fully stuck
            exploration_epsilon = min(1.0, epsilon + boost)
            # Strongly encourage high jump when stuck or approaching obstacle
            encourage_jump = True
            if step % 50 == 0:  # Print warning every 50 steps
                status = "STUCK" if is_stuck else "APPROACHING OBSTACLE"
                print(f"⚠️  {status} at x_pos: {prev_x} for {stuck_counter} steps - Forcing exploration!")
        else:
            exploration_epsilon = epsilon
            # Always encourage some high jump exploration during training
            encourage_jump = epsilon > 0.4  # Phase out high-jump bias as epsilon shrinks
        
        # Early termination check (before taking action)
        if stuck_counter > MAX_STUCK_STEPS:
            print(f"🛑 Episode {episode} terminated early - stuck at x_pos {prev_x} for {stuck_counter} steps")
            # Give large negative reward and push to memory
            shaped_reward = -100.0
            memory.push(state, 0, shaped_reward, state, True)  # Push terminal state
            terminated_due_to_stuck = True
            break
        
        action = select_action(state, policy_net, exploration_epsilon, env, device, encourage_high_jump=encourage_jump)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = preprocess(next_obs)
        done = terminated or truncated

        # --- Reward Shaping ---
        x_progress = info.get('x_pos', prev_x) - prev_x
        score_gain = info.get('score', prev_score) - prev_score
        y_pos = info.get('y_pos', prev_y)
        y_change = prev_y - y_pos  # Positive when Mario goes up (jumps)
        current_x = info.get('x_pos', prev_x)
        forward_velocity = max(0.0, x_progress)
        episode_forward_progress += forward_velocity
        if forward_velocity > 0:
            speed_chain += 1
            max_speed_chain = max(max_speed_chain, speed_chain)
            no_progress_steps = 0
        else:
            speed_chain = 0
            no_progress_steps += 1
        
        # Track maximum y reached (lower y = higher jump)
        if y_pos < max_y_reached:
            max_y_reached = y_pos
        
        # Track jump sequences (keep last 5 actions)
        jump_sequence.append(action)
        if len(jump_sequence) > 5:
            jump_sequence.pop(0)
        
        # Detect if at ground level (y_pos around 79)
        is_at_ground = y_pos >= 77
        if is_at_ground:
            steps_at_ground += 1
            if current_air_chain > max_air_chain:
                max_air_chain = current_air_chain
            current_air_chain = 0
        else:
            steps_at_ground = 0
            air_time_steps += 1
            current_air_chain += 1
        
        # Store previous stuck state BEFORE updating (for unstuck detection)
        was_stuck_before_update = was_stuck_prev_step
        
        # Update stuck counter AFTER getting new info
        if x_progress > 0:
            stuck_counter = 0  # Reset if making progress
            last_progress_x = current_x
        else:
            stuck_counter += 1
        
        # Detect potential obstacle (no progress but not fully stuck yet)
        approaching_obstacle = stuck_counter > 50 and stuck_counter <= STUCK_THRESHOLD
        
        # Re-check if stuck after updating counter (for reward calculation)
        is_stuck_for_reward = stuck_counter > STUCK_THRESHOLD
        
        # Update was_stuck_prev_step for next iteration
        was_stuck_prev_step = is_stuck_for_reward
        
        # ===== IMPROVED REWARD SYSTEM =====
        
        # 1. HIGH JUMP ACTION REWARD (always reward high jumps, more when needed)
        jump_bonus = 0.0
        power_jump_bonus = 0.0
        if action == 2:  # Normal jump (right + A)
            jump_bonus = 3.0  # Base reward for normal jump
            if approaching_obstacle or is_stuck_for_reward:
                jump_bonus = 10.0  # Higher when approaching obstacle
            last_jump_action = 2
        elif action == 4:  # HIGH JUMP (right + A + B) - THIS IS KEY!
            jump_bonus = 50.0  # ALWAYS reward high jump significantly
            if approaching_obstacle:
                jump_bonus *= 2.0 #= 20.0  # Very high when approaching obstacle
            if is_stuck_for_reward:
                jump_bonus *= 5.0  # Maximum when stuck
            last_jump_action = 4
            high_jump_count += 1
            if step % 20 == 0:  # Print occasionally
                print(f"🦘 HIGH JUMP! Reward: {jump_bonus:.1f} (stuck: {stuck_counter}, obstacle: {approaching_obstacle})")
            if y_change > 4:
                power_jump_bonus = 5.0 + y_change * 1.5  # Bonus for strong upward motion
        
        # 2. JUMP HEIGHT REWARD (reward for actual jump height)
        jump_height_bonus = 0.0
        if y_change > 0:  # Mario is going up
            jump_height_bonus = y_change * 6.0  # Increased multiplier (was 4.0)
            # Extra reward for high jumps (y_change > 5 means significant height)
            if y_change > 5:
                jump_height_bonus += 10.0  # Bonus for high jumps
            if y_change > 8:  # Very high jump
                jump_height_bonus += 15.0  # Extra bonus for very high jumps
                print(f"🚀 VERY HIGH JUMP! Height change: {y_change:.1f}, Bonus: {jump_height_bonus:.1f}")
        
        # 3. MAINTAINING JUMP STATE REWARD (reward for staying in air)
        air_time_bonus = 0.0
        air_chain_bonus = 0.0
        if not is_at_ground:  # Mario is in the air
            # Reward being in the air, especially at higher positions
            height_bonus = (79 - y_pos) * 0.5  # Reward being higher up
            air_time_bonus = 1.0 + height_bonus  # Base reward + height bonus
            # Extra reward if we recently did a high jump
            if last_jump_action == 4:
                air_time_bonus *= 2.0  # Double reward for high jump air time
            air_chain_bonus = current_air_chain * 0.3  # Encourage staying airborne longer
        
        # 4. PROGRESS AFTER JUMP REWARD (reward progress made after jumping)
        post_jump_progress_bonus = 0.0
        if last_jump_action in [2, 4] and x_progress > 0:
            # Reward progress made after a jump
            multiplier = 3.0 if last_jump_action == 4 else 2.0  # Higher for high jump
            post_jump_progress_bonus = x_progress * multiplier
            if last_jump_action == 4 and x_progress > 2:
                post_jump_progress_bonus += 10.0  # Extra for clearing obstacle with high jump
                print(f"✅ Progress after HIGH JUMP: {x_progress:.1f}, Bonus: {post_jump_progress_bonus:.1f}, pure_progress: {info.get('x_pos',40)}")
        
        # 5. PROGRESS & SPEED REWARD (favor fast forward motion)
        progress_reward = forward_velocity * forward_velocity #* 2.5  # Stronger weight for moving forward
        #print(f"forward_velocity = {forward_velocity}")
        #sprint_bonus = 0.0
        #if forward_velocity >= 3.0:
        #    sprint_bonus = 5.0  # Extra reward for rapid movement
        #elif forward_velocity >= 1.0:
        #    sprint_bonus = 2.0
        speed_chain_bonus = speed_chain * 0.4  # Encourage sustained progress
        slow_penalty = 0.0
        if no_progress_steps > SLOW_PENALTY_DELAY and is_at_ground:
            slow_penalty = -min(
                MAX_SLOW_PENALTY,
                (no_progress_steps - SLOW_PENALTY_DELAY) * 0.2,
            )  # Penalize lingering on the ground before stuck logic kicks in
        
        # 6. OBSTACLE CLEARING BONUS (reward for clearing obstacles)
        obstacle_clear_bonus = 0.0
        if approaching_obstacle or is_stuck_for_reward:
            if y_pos < 75:  # Mario is high up (jumping over something)
                obstacle_clear_bonus = 5.0  # Reward for being high when near obstacle
            if x_progress > 0 and y_pos < 75:  # Making progress while high
                obstacle_clear_bonus = 15.0  # Large reward for clearing obstacle
                print(f"🏆 OBSTACLE CLEARED! Progress: {x_progress:.1f}, Height: {y_pos:.1f}")
        
        # 7. UNSTUCK BONUS (large reward for breaking out of stuck state)
        unstuck_bonus = 0.0
        # Check if we were stuck before update and now making progress
        if was_stuck_before_update and x_progress > 0:  # Was stuck, now making progress
            unstuck_bonus = 50.0  # Increased from 30.0 to 50.0
            # Extra bonus if high jump was used
            if last_jump_action == 4:
                unstuck_bonus += 25.0  # Extra for high jump unstuck
            print(f"🎉 UNSTUCK! Progress: {x_progress:.1f} (was stuck, jump: {last_jump_action})")
        
        # 8. PREVENTIVE JUMP REWARD (reward jumping before getting stuck)
        preventive_jump_bonus = 0.0
        if approaching_obstacle and action in [2, 4]:
            preventive_jump_bonus = 5.0 if action == 2 else 10.0  # Reward preventive jumping
            if action == 4:
                print(f"🛡️  Preventive HIGH JUMP! (before getting stuck)")
        
        # 9. SURVIVAL BONUS (small reward for staying alive)
        survival_bonus = 0.1 if x_progress > 0 else 0.0

        #10. PURE PROGRESSMENT BONUS
        pure_progress_bonus = 0.0
        if x_progress > 0:
            pure_progress_bonus = 0.5 * info.get("x_pos",40)
        
        # Combine all rewards
        shaped_reward = (reward + 
                progress_reward + 
                #sprint_bonus + 
                speed_chain_bonus + 
                slow_penalty + 
                        0.05 * score_gain + 
                        jump_bonus + 
                        jump_height_bonus + 
                        air_time_bonus + 
                air_chain_bonus + 
                power_jump_bonus + 
                        post_jump_progress_bonus + 
                        obstacle_clear_bonus + 
                        unstuck_bonus + 
                        preventive_jump_bonus + 
                        survival_bonus +
                pure_progress_bonus
                )

        #print(f"shaped rewared = {shaped_reward}")



        # --- Death Penalty ---
        if terminated:
            shaped_reward -= 500.0
        
        # --- Stuck Penalty (escalating penalty the longer stuck) ---
        stuck_penalty = 0.0
        if stuck_counter > 5:  # Start penalizing earlier (after 50 steps)
            # Escalating penalty: -3.0 per 50 steps stuck (more aggressive)
            stuck_penalty = -(stuck_counter // 50) * 3.0
            shaped_reward += stuck_penalty
            # Extra penalty for not jumping when stuck
            if stuck_counter > 100 and action not in [2, 4]:
                shaped_reward -= 2.0  # Penalty for not trying to jump
            if stuck_counter % 50 == 0:  # Print when penalty increases
                print(f"⚠️  Stuck penalty: -{stuck_penalty:.1f} (stuck: {stuck_counter} steps, action: {action})")
        
        # --- Ground Stuck Penalty (penalty for staying at ground when not making progress) ---
        if stuck_counter > 30 and is_at_ground and action not in [2, 4]:
            # Penalty for staying on ground when stuck and not jumping
            ground_stuck_penalty = -1.0
            shaped_reward += ground_stuck_penalty
        
        prev_x = info.get('x_pos', prev_x)
        prev_y = info.get('y_pos', prev_y)
        prev_score = info.get('score', prev_score)
        
        # Reset max_y_reached if back on ground
        if is_at_ground and steps_at_ground > 10:
            max_y_reached = y_pos  # Reset when safely on ground

        memory.push(state, action, shaped_reward, next_state, done)
        state = next_state
        total_reward += shaped_reward

        # --- Training ---
        if len(memory) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            q_values = policy_net(states).gather(1, actions)
            next_q_values = target_net(next_states).max(1)[0].detach()
            expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = torch.nn.functional.mse_loss(q_values.squeeze(), expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            episode_loss_total += loss.item()
            loss_updates += 1

        # Print action names for debugging
        action_names = {0: "NOOP", 1: "RIGHT", 2: "JUMP", 3: "RUN", 4: "HIGH_JUMP"}
        action_name = action_names.get(action, f"UNK({action})")
        # Re-check is_stuck after updating stuck_counter
        is_stuck_now = stuck_counter > STUCK_THRESHOLD
        if step % 100 == 0 or (is_stuck_now and step % 20 == 0):  # Print every 100 steps, or every 20 when stuck
            stuck_info = f" [STUCK: {stuck_counter} steps]" if is_stuck_now else ""
            print(f"Episode {episode} — Step {step} — Action: {action_name}({action}) — Reward: {shaped_reward:.2f} — x_pos: {current_x} — y_pos: {y_pos} — Progress: {x_progress}{stuck_info}")

        if done or step >= MAX_STEPS - 1:
            if current_air_chain > max_air_chain:
                max_air_chain = current_air_chain
            steps_taken = step + 1
            final_x = info.get('x_pos', prev_x)
            avg_reward = total_reward / max(steps_taken, 1)
            avg_speed = episode_forward_progress / max(steps_taken, 1)
            avg_loss = episode_loss_total / max(loss_updates, 1)
            elapsed = time.time() - start_time
            print(
                "✅ Episode {ep} complete — Total Reward: {reward:.2f} — Avg/Step: {avg:.2f} "
                "— Steps: {steps} — Final x: {x_pos} — Forward Prog: {fp:.1f} "
                "— Avg Speed: {avg_speed:.2f} — Avg Loss: {avg_loss:.4f} — Air Steps: {air} "
                "— High Jumps: {hj} — Max Air Chain: {air_chain} — Max Speed Chain: {speed_chain} "
                "— Stuck Count: {stuck} — Stuck Termination: {stuck_term} "
                "— Epsilon: {eps:.3f} — Time: {elapsed:.2f}s".format(
                    ep=episode,
                    reward=total_reward,
                    avg=avg_reward,
                    steps=steps_taken,
                    x_pos=final_x,
                    fp=episode_forward_progress,
                    avg_speed=avg_speed,
                    avg_loss=avg_loss,
                    air=air_time_steps,
                    hj=high_jump_count,
                    air_chain=max_air_chain,
                    speed_chain=max_speed_chain,
                    stuck=stuck_counter,
                    stuck_term=terminated_due_to_stuck,
                    eps=epsilon,
                    elapsed=elapsed,
                )
            )
            break

    # --- Save Best Model with Timestamp ---
    if total_reward > best_reward:
        best_reward = total_reward
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_path = f"checkpoints/best_model_{timestamp}.pt"
        torch.save(policy_net.state_dict(), best_model_path)
        print(f"💾 New best model saved: {best_model_path} with reward {best_reward:.2f}")
        try:
            reward_file = open(BEST_REWARD_PATH, "w")
            reward_file.write(str(best_reward))
            reward_file.close()
        except Exception as e:
            printf("Could not write best score to file.")
            

    # --- Epsilon Decay ---
    epsilon_history.append(epsilon)
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    # --- Target Network Update ---
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # --- Save Checkpoint ---
    if episode % 50 == 0:
        torch.save(policy_net.state_dict(), SAVE_PATH)

    steps_recorded = min(step + 1, MAX_STEPS)
    avg_speed_episode = episode_forward_progress / max(steps_recorded, 1)
    avg_loss_episode = episode_loss_total / max(loss_updates, 1)
    forward_progress_history.append(episode_forward_progress)
    avg_speed_history.append(avg_speed_episode)
    air_steps_history.append(air_time_steps)
    high_jump_history.append(high_jump_count)
    max_air_chain_history.append(max_air_chain)
    loss_history.append(avg_loss_episode)
    reward_history.append(total_reward)
    stuck_termination_history.append(1 if terminated_due_to_stuck else 0)

# --- Finalize ---
if viewer is not None:
    viewer.close()
    print("👁️  Visual display closed")
if video_enabled and video is not None:
    video.release()
    print(f"📹 Video saved to {VIDEO_PATH}")
env.close()
plot_rewards(
    {
        "reward": reward_history,
        "forward_progress": forward_progress_history,
        "avg_speed": avg_speed_history,
        "air_steps": air_steps_history,
        "high_jumps": high_jump_history,
        "max_air_chain": max_air_chain_history,
        "avg_loss": loss_history,
        "epsilon": epsilon_history,
        "stuck_termination": stuck_termination_history,
    },
    REWARD_PLOT_PATH,
)
print(f"📉 Episodes ending due to stuck detection: {sum(stuck_termination_history)}")
print(f"🏁 Training complete. Reward plot saved to {REWARD_PLOT_PATH}")
