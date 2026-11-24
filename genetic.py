import pandas as pd
import numpy as np
import random
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(BASE_DIR, "GA_feature_selection")

MIN_FEATURES = 30
MAX_FEATURES = 46
POPULATION_SIZE = 40
GENERATIONS = 80
MUTATION_RATE = 0.03
CROSSOVER_RATE = 0.80
ELITE_COUNT = 2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ==================== LOOP FOR SEED 1 TO 5 ====================
for SEED in [1,2,3,4,5]:
    print(f"\n\n########## GA - SEED {SEED} ##########")

    INPUT_FILE = os.path.join(BASE_DIR, "data", f"ciciot_minmax_seed{SEED}.csv")
    OUTPUT_FOLDER = os.path.join(OUTPUT_BASE, f"seed{SEED}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Loading dataset...", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)
    print("Original dataset shape:", df.shape)

    # FIX: Drop BOTH label and Sub_Category from X
    label_column = next((col for col in df.columns if col.lower() == "label"), None)
    sub_column = next((col for col in df.columns if col.lower() == "sub_category"), None)

    cols_to_drop = []
    if label_column is not None:
        cols_to_drop.append(label_column)
    if sub_column is not None:
        cols_to_drop.append(sub_column)

    labels = df[label_column].copy() if label_column is not None else None
    sub_labels = df[sub_column].copy() if sub_column is not None else None

    X = df.drop(columns=cols_to_drop)
    print(f"Dropped target cols: {cols_to_drop}")

    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        print("Removing constant features:", constant_features)
        X = X.drop(columns=constant_features)

    feature_names = X.columns.tolist()
    number_of_features = len(feature_names)
    print("Number of usable input features:", number_of_features)

    if number_of_features < MIN_FEATURES:
        raise ValueError("There are fewer usable features than MIN_FEATURES.")

    MAX_FEATURES_CURR = min(MAX_FEATURES, number_of_features)

    print("\nCalculating feature statistics...")
    correlation_matrix = X.corr().abs().fillna(0).to_numpy()
    variances = X.var().to_numpy()
    max_variance = variances.max()
    normalized_variance = variances / max_variance if max_variance > 0 else np.zeros_like(variances)

    def calculate_fitness(chromosome):
        selected_indices = np.where(chromosome == 1)[0]
        selected_count = len(selected_indices)
        if selected_count < MIN_FEATURES or selected_count > MAX_FEATURES_CURR:
            return -1
        selected_corr = correlation_matrix[np.ix_(selected_indices, selected_indices)]
        redundancy = selected_corr[np.triu_indices(selected_count, k=1)].mean() if selected_count > 1 else 0
        diversity = normalized_variance[selected_indices].mean()
        size_penalty = (selected_count - MIN_FEATURES) / max(1, MAX_FEATURES_CURR - MIN_FEATURES)
        return 0.60 * (1 - redundancy) + 0.35 * diversity - 0.05 * size_penalty

    def create_chromosome():
        chromosome = np.zeros(number_of_features, dtype=int)
        selected = random.sample(range(number_of_features), random.randint(MIN_FEATURES, MAX_FEATURES_CURR))
        chromosome[selected] = 1
        return chromosome

    population = [create_chromosome() for _ in range(POPULATION_SIZE)]

    def tournament_selection(population):
        return max(random.sample(population, 3), key=calculate_fitness).copy()

    def crossover(parent1, parent2):
        if random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, number_of_features - 1)
        return np.concatenate([parent1[:point], parent2[point:]]), np.concatenate([parent2[:point], parent1[point:]])

    def repair_chromosome(chromosome):
        selected = np.where(chromosome == 1)[0].tolist()
        unselected = np.where(chromosome == 0)[0].tolist()
        while len(selected) < MIN_FEATURES:
            chosen = random.choice(unselected)
            chromosome[chosen] = 1
            selected.append(chosen); unselected.remove(chosen)
        while len(selected) > MAX_FEATURES_CURR:
            chosen = random.choice(selected)
            chromosome[chosen] = 0
            selected.remove(chosen)
        return chromosome

    def mutate(chromosome):
        for i in range(number_of_features):
            if random.random() < MUTATION_RATE:
                chromosome[i] = 1 - chromosome[i]
        return repair_chromosome(chromosome)

    print("\nStarting Genetic Algorithm...\n")
    best_fitness_history = []
    for generation in range(GENERATIONS):
        population = sorted(population, key=calculate_fitness, reverse=True)
        best_fitness = calculate_fitness(population[0])
        best_fitness_history.append(best_fitness)
        print(f"Gen {generation+1}/{GENERATIONS} | Seed {SEED} | Best: {best_fitness:.5f} | Selected: {int(population[0].sum())}")
        new_population = [c.copy() for c in population[:ELITE_COUNT]]
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))
        population = new_population

    best_chromosome = max(population, key=calculate_fitness)
    selected_features = [feature_names[i] for i in np.where(best_chromosome == 1)[0]]

    print("\n========== FINAL RESULTS SEED", SEED, "==========")
    print("Original usable features:", number_of_features)
    print("Selected features:", len(selected_features))
    print("Best fitness:", calculate_fitness(best_chromosome))
    for n, f in enumerate(selected_features, start=1):
        print(n, f)

    pd.DataFrame({"Selected_Feature": selected_features}).to_csv(os.path.join(OUTPUT_FOLDER, f"GA_selected_features_seed{SEED}.csv"), index=False)

    reduced_df = X[selected_features].copy()
    if labels is not None:
        reduced_df[label_column] = labels
    if sub_labels is not None:
        reduced_df[sub_column] = sub_labels
    reduced_df.to_csv(os.path.join(OUTPUT_FOLDER, f"GA_reduced_dataset_seed{SEED}.csv"), index=False)

    pd.DataFrame({"Generation": range(1, GENERATIONS + 1), "Best_Fitness": best_fitness_history}).to_csv(os.path.join(OUTPUT_FOLDER, f"GA_fitness_history_seed{SEED}.csv"), index=False)

    print(f"\nSeed {SEED} saved to: {OUTPUT_FOLDER}")

print("\n\nALL 5 SEEDS DONE!")