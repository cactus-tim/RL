import os
import sys
from pathlib import Path
import numpy as np
from sumo_rl import SumoEnvironment
from deep_q_agent import DeepQAgent


def create_environment(output_file_prefix, random_seed):
    script_dir = Path(__file__).resolve().parent

    custom_net = script_dir / "big-intersection.net.xml"
    custom_routes = script_dir / "routes.rou.xml"

    if not custom_net.exists():
        custom_net = Path.cwd() / "big-intersection.net.xml"
        custom_routes = Path.cwd() / "routes.rou.xml"

    if custom_net.exists() and custom_routes.exists():
        net_file = str(custom_net)
        route_file = str(custom_routes)

    return SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        single_agent=True,
        out_csv_name=output_file_prefix,
        use_gui=False,
        num_seconds=3600,
        yellow_time=4,
        min_green=5,
        max_green=50,
        sumo_seed=random_seed,
    )


def train_dqn_agent(num_runs=2, total_timesteps=5000):
    script_dir = Path(__file__).resolve().parent
    output_directory = script_dir / "outputs" / "custom_dqn"
    output_directory.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("DQN АГЕНТ")
    print("="*70)

    for run_idx in range(1, num_runs + 1):
        print(f"\n{'='*70}")
        print(f"Запуск {run_idx}/{num_runs}")
        print(f"{'='*70}")

        run_seed = 100 + run_idx

        output_prefix = str(output_directory / f"run{run_idx}")

        environment = create_environment(
            output_file_prefix=output_prefix,
            random_seed=run_seed
        )

        initial_observation, info = environment.reset(seed=run_seed)

        observation_dim = int(np.array(initial_observation).shape[0])
        num_actions = int(environment.action_space.n)

        print(f"  Размерность состояния: {observation_dim}")
        print(f"  Количество действий: {num_actions}")

        dqn_agent = DeepQAgent(
            state_size=observation_dim,
            action_size=num_actions,
            hidden_layers=(64, 64),
            learning_rate=1e-3,
            discount_factor=0.95,
            exploration_rate=1.0,
            exploration_decay=0.995,
            exploration_min=0.05,
            memory_size=5000,
            training_batch_size=32,
            target_network_update_frequency=100,
            enable_double_dqn=True,
            enable_prioritized_replay=False,
        )

        episode_count = 0
        episode_reward = 0
        episode_steps = 0
        current_observation = initial_observation

        for timestep in range(total_timesteps):
            selected_action = dqn_agent.select_action(current_observation)

            next_observation, step_reward, terminated, truncated, info = environment.step(selected_action)

            episode_done = bool(terminated or truncated)

            dqn_agent.remember(
                current_observation,
                selected_action,
                step_reward,
                next_observation,
                episode_done
            )

            if timestep % 2 == 0 and timestep > 100:
                dqn_agent.learn_from_experience()

            episode_reward += step_reward
            episode_steps += 1

            current_observation = next_observation

            if episode_done:
                episode_count += 1
                print(f"    Эпизод {episode_count} завершён - "
                      f"Награда: {episode_reward:.2f}, "
                      f"Шагов: {episode_steps}, "
                      f"Timestep: {timestep+1}")

                current_observation, info = environment.reset()
                episode_reward = 0
                episode_steps = 0

            if (timestep + 1) % 1000 == 0:
                buffer_size = dqn_agent.replay_buffer.size()
                exploration = dqn_agent.epsilon

                print(f"\n  [Прогресс] Шаг {timestep+1}/{total_timesteps}")
                print(f"    Epsilon: {exploration:.4f}")
                print(f"    Размер буфера: {buffer_size}")
                print(f"    Завершено эпизодов: {episode_count}")

        environment.close()

        print(f"\n{'='*70}")
        print(f"Запуск {run_idx}/{num_runs} завершён")
        print(f"  Всего эпизодов: {episode_count}")
        print(f"{'='*70}")


if __name__ == "__main__":
    train_dqn_agent(num_runs=2, total_timesteps=50000)
