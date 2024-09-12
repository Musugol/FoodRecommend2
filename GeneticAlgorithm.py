from geneticalgorithm import geneticalgorithm as ga
import numpy as np

def optimizeWithGeneticAlgorithm(filteredData, dailyTargets):
    """
    유전 알고리즘을 사용하여 최적의 음식 조합 찾기

    Parameters:
    filteredData (pd.DataFrame): 필터링된 음식 데이터
    dailyTargets (dict): 하루 목표 섭취량(칼로리, 단백질, 지방, 탄수화물)

    Returns:
    list: 최적의 음식 조합 인덱스
    """
    def objectiveFunction(selection):
        totalKcal = np.sum(selection * filteredData['kcal_std'].values)
        totalProtein = np.sum(selection * filteredData['protein_std'].values)
        totalFat = np.sum(selection * filteredData['fat_std'].values)
        totalCarb = np.sum(selection * filteredData['carb_std'].values)

        kcalDiff = abs(totalKcal - dailyTargets['kcal'])
        proteinDiff = abs(totalProtein - dailyTargets['protein'])
        fatDiff = abs(totalFat - dailyTargets['fat'])
        carbDiff = abs(totalCarb - dailyTargets['carb'])

        return kcalDiff + proteinDiff + fatDiff + carbDiff
    
    varBound = np.array([[0,1]] * len(filteredData))

    algorithmParam = {
        'max_num+iteration': 1000,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_imporov': 100
    }

    model = ga(
        function= objectiveFunction,
        dimension=len(filteredData),
        variable_type='binary',
        variable_boundaries=varBound,
        algorithm_parameters=algorithmParam
    )
    model.run()
    return model.output_dict['variable']