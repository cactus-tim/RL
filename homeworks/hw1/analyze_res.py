import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def analyze_qlearning_results(output_dir):
    output_path = Path(output_dir)

    csv_files = sorted(list(output_path.glob("*.csv")))

    print(f"\n{'=' * 70}")
    print(output_path.name)
    print(f"{'=' * 70}")
    print(f"Файлов: {len(csv_files)}")

    runs = {}
    for csv_file in csv_files:
        parts = csv_file.stem.split('_')
        if len(parts) >= 3:
            run_num = parts[0]
            episode_num = parts[2]

            if run_num not in runs:
                runs[run_num] = {}
            runs[run_num][episode_num] = csv_file

    for run_name in sorted(runs.keys()):
        print(f"\n{run_name.upper()}:")
        episodes = runs[run_name]

        last_episode = sorted(episodes.keys())[-1]
        last_file = episodes[last_episode]

        print(f"  Эпизодов: {len(episodes)}")
        print(f"  Последний эпизод: {last_episode}")

        df = pd.read_csv(last_file)

        if run_name == list(runs.keys())[0] and last_episode == sorted(episodes.keys())[-1]:
            print(f"\n   Метрики: {len(df.columns)}")

        print(f"\n  Статистика последнего эпизода ({last_episode}):")

        if 'system_mean_waiting_time' in df.columns:
            print(f"    Среднее время ожидания: {df['system_mean_waiting_time'].mean():.2f} сек")

        if 'system_total_stopped' in df.columns:
            print(f"    Средние остановки: {df['system_total_stopped'].mean():.2f}")

        print(f"    Общее количество шагов: {len(df)}")


def analyze_dqn_results(output_dir):
    output_path = Path(output_dir)

    csv_files = sorted(list(output_path.glob("*.csv")))

    print(f"\n{'=' * 70}")
    print(f"DQN: {output_path.name}")
    print(f"{'=' * 70}")
    print(f"Файлы: {len(csv_files)}")

    last_file = csv_files[-1]
    print(f"Файл: {last_file.name}")

    df = pd.read_csv(last_file)

    print(f"\nСтатистика:")
    if 'system_mean_waiting_time' in df.columns:
        print(f"  Среднее время ожидания: {df['system_mean_waiting_time'].mean():.2f} сек")
        print(f"  Максимум: {df['system_mean_waiting_time'].max():.2f} сек")
        print(f"  Минимум: {df['system_mean_waiting_time'].min():.2f} сек")

    print(f"  Шагов симуляции: {len(df)}")

    return df


def plot_dqn_analysis(dqn_dir, save_path=None):
    dqn_path = Path(dqn_dir)

    csv_files = sorted(list(dqn_path.glob("*.csv")))

    df = pd.read_csv(csv_files[-1])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Analysis', fontsize=16, fontweight='bold')

    if 'system_mean_waiting_time' in df.columns:
        ax = axes[0, 0]
        ax.plot(df['system_mean_waiting_time'], linewidth=2, color='#E74C3C')
        ax.axhline(y=df['system_mean_waiting_time'].mean(),
                   color='blue', linestyle='--', label=f'Среднее: {df["system_mean_waiting_time"].mean():.2f}')
        ax.set_xlabel('Шаг симуляции')
        ax.set_ylabel('Среднее время ожидания (сек)')
        ax.set_title('Динамика времени ожидания')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if 'system_total_stopped' in df.columns:
        ax = axes[0, 1]
        ax.plot(df['system_total_stopped'], linewidth=2, color='#3498DB')
        ax.axhline(y=df['system_total_stopped'].mean(),
                   color='red', linestyle='--', label=f'Среднее: {df["system_total_stopped"].mean():.2f}')
        ax.set_xlabel('Шаг симуляции')
        ax.set_ylabel('Количество остановленных машин')
        ax.set_title('Остановленные машины')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if 'system_mean_waiting_time' in df.columns:
        ax = axes[1, 0]
        window = min(50, max(5, len(df) // 10))
        if window > 1:
            rolling_mean = df['system_mean_waiting_time'].rolling(window=window).mean()
            ax.plot(rolling_mean, linewidth=2, color='#2ECC71', label=f'Скользящее среднее ({window})')
            ax.plot(df['system_mean_waiting_time'], alpha=0.3, color='gray', label='Исходные')
        else:
            ax.plot(df['system_mean_waiting_time'], linewidth=2, color='#2ECC71')
        ax.set_xlabel('Шаг симуляции')
        ax.set_ylabel('Время ожидания (сек)')
        ax.set_title('Сглаженная динамика')
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')

    stats = [
        ['Среднее время ожидания', f"{df['system_mean_waiting_time'].mean():.2f} сек"],
        ['Максимум', f"{df['system_mean_waiting_time'].max():.2f} сек"],
        ['Минимум', f"{df['system_mean_waiting_time'].min():.2f} сек"],
    ]

    if 'system_total_stopped' in df.columns:
        stats.append(['Средние остановки', f"{df['system_total_stopped'].mean():.2f}"])

    stats.append(['Шагов симуляции', f"{len(df)}"])

    table = ax.table(
        cellText=stats,
        colLabels=['Метрика', 'Значение'],
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for i in range(2):
        table[(0, i)].set_facecolor('#E74C3C')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_comparison(baseline_dir, custom_dir, dqn_dir, save_path=None):
    baseline_path = Path(baseline_dir)
    custom_path = Path(custom_dir)
    dqn_path = Path(dqn_dir)

    baseline_files = sorted(list(baseline_path.glob("*_ep*.csv")))
    custom_files = sorted(list(custom_path.glob("*_ep*.csv")))
    dqn_files = sorted(list(dqn_path.glob("*.csv"))) if dqn_path.exists() else []

    df_baseline = pd.read_csv(baseline_files[-1])
    df_custom = pd.read_csv(custom_files[-1])
    df_dqn = pd.read_csv(dqn_files[-1]) if dqn_files else None

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_baseline['system_mean_waiting_time'],
             label='Baseline Q-Learning', alpha=0.7, linewidth=2, color='#3498DB')
    ax1.plot(df_custom['system_mean_waiting_time'],
             label='Custom Q-Learning', alpha=0.7, linewidth=2, color='#2ECC71')

    if df_dqn is not None:
        dqn_data = df_dqn['system_mean_waiting_time']
        ax1.plot(dqn_data,
                 label=f'DQN ({len(df_dqn)} шагов)', alpha=0.7, linewidth=2, color='#E74C3C')

    ax1.set_xlabel('Шаг симуляции', fontsize=12)
    ax1.set_ylabel('Среднее время ожидания (сек)', fontsize=12)
    ax1.set_title('Время ожидания', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df_baseline['system_total_stopped'],
             label='Baseline', alpha=0.7, linewidth=2, color='#3498DB')
    ax2.plot(df_custom['system_total_stopped'],
             label='Custom', alpha=0.7, linewidth=2, color='#2ECC71')
    ax2.set_xlabel('Шаг симуляции')
    ax2.set_ylabel('Остановленные машины')
    ax2.set_title('Q-Learning: Остановки')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. DQN отдельно
    ax3 = fig.add_subplot(gs[1, 1])
    if df_dqn is not None:
        ax3.plot(df_dqn['system_total_stopped'],
                 linewidth=2, color='#E74C3C')
        ax3.set_xlabel('Шаг симуляции')
        ax3.set_ylabel('Остановленные машины')
        ax3.set_title('DQN: Остановки')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'DQN данные\nне найдены',
                 ha='center', va='center', fontsize=14, color='gray')
        ax3.axis('off')

    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    metrics = [
        ['Метрика', 'Baseline Q-L', 'Custom Q-L', 'DQN'],
        ['Среднее время ожидания (сек)',
         f"{df_baseline['system_mean_waiting_time'].mean():.2f}",
         f"{df_custom['system_mean_waiting_time'].mean():.2f}",
         f"{df_dqn['system_mean_waiting_time'].mean():.2f}" if df_dqn is not None else 'N/A'],
        ['Средние остановки',
         f"{df_baseline['system_total_stopped'].mean():.2f}",
         f"{df_custom['system_total_stopped'].mean():.2f}",
         f"{df_dqn['system_total_stopped'].mean():.2f}" if df_dqn is not None else 'N/A'],
        ['Шагов симуляции',
         f"{len(df_baseline)}",
         f"{len(df_custom)}",
         f"{len(df_dqn)}" if df_dqn is not None else 'N/A']
    ]

    table = ax4.table(
        cellText=metrics,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.2, 0.8, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for i in range(4):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')

    baseline_wait = df_baseline['system_mean_waiting_time'].mean()
    custom_wait = df_custom['system_mean_waiting_time'].mean()

    if custom_wait < baseline_wait:
        table[(1, 2)].set_facecolor('#D5F4E6')
    else:
        table[(1, 1)].set_facecolor('#D5F4E6')

    plt.suptitle('Полное сравнение методов обучения',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def main():
    script_dir = Path(__file__).resolve().parent
    outputs_dir = script_dir / "outputs"

    baseline_dir = outputs_dir / "baseline"
    custom_dir = outputs_dir / "custom"
    dqn_dir = outputs_dir / "custom_dqn"

    if baseline_dir.exists():
        analyze_qlearning_results(baseline_dir)

    if custom_dir.exists():
        analyze_qlearning_results(custom_dir)

    if dqn_dir.exists():
        analyze_dqn_results(dqn_dir)

    if dqn_dir.exists():
        plot_dqn_analysis(dqn_dir, save_path=outputs_dir / "dqn_analysis.png")

    if baseline_dir.exists() and custom_dir.exists():
        plot_all_comparison(
            baseline_dir,
            custom_dir,
            dqn_dir,
            save_path=outputs_dir / "full_comparison.png"
        )


if __name__ == "__main__":
    main()
