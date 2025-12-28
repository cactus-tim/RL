import os
import sys
from pathlib import Path
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
from q_learning_agent import QLearningAgent
import numpy as np


def get_sumo_files():
    import pkg_resources
    net_file = pkg_resources.resource_filename(
        'sumo_rl', 'nets/4x4-Lucas/4x4.net.xml'
    )
    route_file = pkg_resources.resource_filename(
        'sumo_rl', 'nets/4x4-Lucas/4x4c1c2c1c2.rou.xml'
    )
    return net_file, route_file


def train_qlearning(agent_variant, num_runs=2, num_episodes=10):
    net_file, route_file = get_sumo_files()
    
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=20000,
        min_green=5,
        delta_time=5,
    )
    
    script_dir = Path(__file__).resolve().parent
    output_directory = script_dir / "outputs" / agent_variant
    output_directory.mkdir(parents=True, exist_ok=True)
    
    for run_number in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"Запуск {run_number}/{num_runs} ({agent_variant})")
        print(f"{'='*60}")
        
        initial_observations = env.reset(seed=run_number)
        
        traffic_agents = {}
        
        for traffic_light_id in env.ts_ids:
            encoded_state = env.encode(initial_observations[traffic_light_id], traffic_light_id)
            
            if agent_variant == "baseline":
                traffic_agents[traffic_light_id] = QLAgent(
                    starting_state=encoded_state,
                    state_space=env.observation_space,
                    action_space=env.action_space,
                    alpha=0.1,
                    gamma=0.95,
                    exploration_strategy=EpsilonGreedy(
                        initial_epsilon=0.5, 
                        min_epsilon=0.01, 
                        decay=0.9995
                    ),
                )
            else:
                traffic_agents[traffic_light_id] = QLearningAgent(
                    initial_state=encoded_state,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    learning_rate=0.1,
                    discount_factor=0.95,
                    exploration_rate=0.5,
                    exploration_decay=0.9995,
                    exploration_min=0.01,
                )
        
        env.save_csv(str(output_directory / f"run{run_number}"), run_number)
        
        for episode_num in range(1, num_episodes + 1):
            if episode_num != 1:
                episode_seed = run_number * 10000 + episode_num
                initial_observations = env.reset(seed=episode_seed)
                
                for ts_id in initial_observations.keys():
                    new_state = env.encode(initial_observations[ts_id], ts_id)
                    if agent_variant == "baseline":
                        traffic_agents[ts_id].state = new_state
                    else:
                        traffic_agents[ts_id].current_state = new_state

            episode_finished = {"__all__": False}
            step_count = 0

            while not episode_finished["__all__"]:
                if agent_variant == "baseline":
                    agent_actions = {
                        ts_id: traffic_agents[ts_id].act()
                        for ts_id in traffic_agents.keys()
                    }
                else:
                    agent_actions = {
                        ts_id: traffic_agents[ts_id].select_action()
                        for ts_id in traffic_agents.keys()
                    }

                observations, rewards, episode_finished, info = env.step(action=agent_actions)

                for agent_id in observations.keys():
                    encoded_next_state = env.encode(observations[agent_id], agent_id)
                    agent_reward = rewards[agent_id]
                    is_done = episode_finished.get(agent_id, False)

                    if agent_variant == "baseline":
                        traffic_agents[agent_id].learn(
                            next_state=encoded_next_state,
                            reward=agent_reward,
                            done=is_done
                        )
                    else:
                        traffic_agents[agent_id].update(
                            next_state=encoded_next_state,
                            reward=agent_reward,
                            is_terminal=is_done
                        )

                step_count += 1

            if agent_variant == "custom":
                avg_epsilon = np.mean([
                    agent.get_exploration_rate()
                    for agent in traffic_agents.values()
                ])
                print(f"  Эпизод {episode_num}/{num_episodes} - "
                      f"Шагов: {step_count}, "
                      f"Средний epsilon: {avg_epsilon:.4f}")
            else:
                print(f"  Эпизод {episode_num}/{num_episodes} - "
                      f"Шагов: {step_count}")

        env.save_csv(str(output_directory / f"run{run_number}"), run_number)

        if agent_variant == "custom":
            total_states = sum(
                agent.get_q_table_size()
                for agent in traffic_agents.values()
            )
            print(f"\n  Общее количество изученных состояний: {total_states}")
        else:
            print(f"\n  Обучение завершено")

    env.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Q-LEARNING АГЕНТЫ")
    print("="*60)

    print("\n[1/2] BASELINE агент")
    train_qlearning("baseline", num_runs=2, num_episodes=10)

    print("\n[2/2] CUSTOM агент")
    train_qlearning("custom", num_runs=2, num_episodes=10)
