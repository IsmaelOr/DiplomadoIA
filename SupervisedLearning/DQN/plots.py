
import matplotlib.pyplot as plt
import numpy as np

def moving_average(x, w=50):
    if len(x) < w:
        return np.array(x, dtype=float)
    x = np.array(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training(episode_rewards, episode_epsilons, losses, title_prefix="DQN"):
    # 1) Rewards por episodio + media móvil
    plt.figure()
    plt.plot(episode_rewards, label="Reward por episodio")
    ma = moving_average(episode_rewards, w=50)
    plt.plot(range(len(episode_rewards) - len(ma) + 1, len(episode_rewards) + 1), ma, label="Media móvil (50)")
    plt.title(f"Entrenamiento {title_prefix}: Recompensa")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    # 2) Epsilon por episodio
    plt.figure()
    plt.plot(episode_epsilons)
    plt.title(f"{title_prefix} - Exploración (epsilon) por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Epsilon")
    plt.grid(True)

    # 3) Loss a lo largo de las actualizaciones
    if len(losses) > 0:
        plt.figure()
        plt.plot(losses, label="Loss (actualizaciones)")
        ma_l = moving_average(losses, w=200)
        plt.plot(range(len(losses) - len(ma_l) + 1, len(losses) + 1), ma_l, label="Media móvil (200)")
        plt.title(f"{title_prefix} - Loss durante el entrenamiento")
        plt.xlabel("Actualización")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

    plt.show()