import numpy as np
import random

def optimizeWithGeneticAlgorithm(filteredData, mealTargets, tolerance=0.1, min_items=3, max_items=3):
    """
    유전 알고리즘을 사용하여 최적의 음식 조합 찾기.
    개체는 선택된 음식의 인덱스 리스트로 표현됩니다.
    """

    num_foods = len(filteredData)

    def objectiveFunction(individual):
        selected_items = filteredData.iloc[individual]

        totalKcal = np.sum(selected_items['kcal'].values)

        # 목표 칼로리와의 차이 계산
        kcalDiff = abs(totalKcal - mealTargets['kcal'])

        # 칼로리가 목표 범위 내에 있는지 확인
        penalty = 0
        if not (mealTargets['kcal'] * (1 - tolerance) <= totalKcal <= mealTargets['kcal'] * (1 + tolerance)):
            penalty += 1000  # 페널티 값을 적절히 조정

        return kcalDiff + penalty

    def initialize_population(pop_size):
        """
        초기 집단 생성: 각 개체는 음식의 인덱스 리스트입니다.
        """
        population = []
        for _ in range(pop_size):
            num_items = random.randint(min_items, max_items)
            individual = random.sample(range(num_foods), num_items)
            population.append(individual)
        return population

    def evaluate_population(population):
        """
        인구의 각 개체에 대해 적합도 평가를 수행하고, 적합도 목록을 반환.
        """
        fitness_values = []
        for individual in population:
            fitness = objectiveFunction(individual)
            fitness_values.append(fitness)
        return np.array(fitness_values)

    def selection(population, fitness_values):
        """
        토너먼트 선택 방식으로 부모 선택.
        """
        idx1, idx2 = random.sample(range(len(population)), 2)
        if fitness_values[idx1] < fitness_values[idx2]:
            return population[idx1]
        else:
            return population[idx2]

    def crossover(parent1, parent2):
        """
        두 부모의 유전 정보를 교차시켜 자식 생성.
        """
        child = list(set(parent1) | set(parent2))  # 부모의 음식을 합침
        random.shuffle(child)
        num_items = random.randint(min_items, max_items)
        child = child[:num_items]  # 음식 개수 제한 적용
        return child

    def mutate(individual):
        """
        변이 과정: 음식 교체, 추가 또는 제거.
        """
        mutation_type = random.choice(['swap', 'add', 'remove'])
        if mutation_type == 'swap' and len(individual) > 0:
            idx = random.randint(0, len(individual) - 1)
            new_food = random.randint(0, num_foods - 1)
            individual[idx] = new_food
        elif mutation_type == 'add' and len(individual) < max_items:
            new_food = random.randint(0, num_foods - 1)
            if new_food not in individual:
                individual.append(new_food)
        elif mutation_type == 'remove' and len(individual) > min_items:
            idx = random.randint(0, len(individual) - 1)
            del individual[idx]
        return individual

    def evolve(population, fitness_values, mutation_probability, elitism=True):
        """
        다음 세대 생성.
        """
        new_population = []
        if elitism:
            # 최적 해를 다음 세대에 그대로 전달 (엘리트 선택)
            best_individual = population[np.argmin(fitness_values)]
            new_population.append(best_individual.copy())

        while len(new_population) < len(population):
            parent1 = selection(population, fitness_values)
            parent2 = selection(population, fitness_values)

            child = crossover(parent1, parent2)

            if random.random() < mutation_probability:
                child = mutate(child)

            new_population.append(child)

        return new_population

    def on_generation(generation, population, fitness_values):
        """
        세대별 최적의 적합도와 선택된 음식 출력.
        """
        best_fitness = np.min(fitness_values)
        best_index = np.argmin(fitness_values)
        best_individual = population[best_index]
        selected_items = filteredData.iloc[best_individual]
        total_kcal = np.sum(selected_items['kcal'].values)

        print(f"세대 {generation}: 최적 적합도 {best_fitness}, 선택된 음식들: {selected_items['food_name'].tolist()}, 총 칼로리: {total_kcal}")

    # 유전 알고리즘 파라미터 설정
    algorithmParam = {
        'max_num_iteration': 100,
        'population_size': 50,
        'mutation_probability': 0.5,
        'elit_ratio': 0.02,
        'max_iteration_without_improv': 10
    }

    # 초기 집단 생성
    population = initialize_population(algorithmParam['population_size'])

    best_fitness = float('inf')
    no_improv = 0

    # 각 세대별 상태를 출력
    for generation in range(algorithmParam['max_num_iteration']):
        # 인구의 적합도 평가
        fitness_values = evaluate_population(population)

        # 세대별 출력
        on_generation(generation, population, fitness_values)

        current_best_fitness = np.min(fitness_values)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            no_improv = 0
        else:
            no_improv += 1

        if best_fitness <= 0 or no_improv >= algorithmParam['max_iteration_without_improv']:
            print("최적 해가 발견되었거나 개선이 없어 종료합니다.")
            break

        # 다음 세대 생성
        population = evolve(population, fitness_values, algorithmParam['mutation_probability'])

    # 최적의 인덱스 반환
    best_individual = population[np.argmin(fitness_values)]
    optimized_indices = best_individual
    return optimized_indices
