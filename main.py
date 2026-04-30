from training.run_nsga3 import run_nsga3


if __name__ == "__main__":
    run_nsga3(
        env_id="mo-halfcheetah-v5",
        seed=0,
        population_size=16,
        generations=10,
        eval_episodes=1,
        max_steps=300,
        hidden_dim=64,
        crossover_rate=0.9,
        mutation_rate=0.2,
        mutation_sigma=0.05,
        output_dir="outputs",
    )