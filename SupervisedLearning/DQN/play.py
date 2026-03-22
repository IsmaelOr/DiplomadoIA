import torch
import gymnasium as gym
import time
from DQNIsOrEs import DQNCNN, make_env, obs_to_tensor
from configIsOrEs import Config

def play_game_steps(max_steps=1000):
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(cfg.env_id, 701, True)

    n_actions = env.action_space.n

    model = DQNCNN(n_actions=n_actions, in_channels=cfg.atari_frame_stack).to(device)
    model.load_state_dict(torch.load("dqn_atari_model.pth", map_location=device))
    model.eval()

    obs, _ = env.reset()

    done = False
    step = 0

    while step < max_steps:

        with torch.no_grad():
            st = obs_to_tensor(obs, device, atari=True)
            qvals = model(st)
            action = int(torch.argmax(qvals, dim=1).item())

        obs, reward, terminated, truncated, info = env.step(action)

        step += 1

        time.sleep(0.02)

    print("Episodio terminado")

    env.close()

def play_game():
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(cfg.env_id, cfg.seed, render=True)

    n_actions = env.action_space.n

    model = DQNCNN(n_actions=n_actions, in_channels=cfg.atari_frame_stack).to(device)
    model.load_state_dict(torch.load("dqn_atari_model.pth", map_location=device))
    model.eval()

    obs, _ = env.reset()

    done = False

    while not done:

        with torch.no_grad():
            st = obs_to_tensor(obs, device, atari=True)
            qvals = model(st)
            action = int(torch.argmax(qvals, dim=1).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()