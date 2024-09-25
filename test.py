from DataBase.DatabaseHandler import DatabaseHandler
from DataProcessing import Processing
from UserProcessing import UserProfile, createUserVector
from Filtering.ContentBasedFiltering import contentBasedFiltering
from Filtering.CollaborativeFiltering import collaborativeFiltering
from GeneticAlgorithm import optimizeWithGeneticAlgorithm
import numpy as np
import pandas as pd

# 무한대와 NaN 값 처리 함수 정의
def cleanNumericData(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = df.clip(lower=-1e10, upper=1e10)
    return df

if __name__ == '__main__':
    # MySQL DB 연결 설정
    dbConfig = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': '785466',
        'database': 'food_data',
        'port': 3306,
    }

    # DB 연결(인스턴스 생성)
    dbHandler = DatabaseHandler(dbConfig)

    # food_data 및 feedback 데이터 로드
    foodData = dbHandler.loadData("SELECT * FROM food_data")
    user_id = 'k65654'
    feedbackData = dbHandler.loadData(f"SELECT user_id, food_code, rating FROM feedback WHERE user_id = '{user_id}'")
    print(feedbackData.columns)

    # 100g 기준 영양소 정보 추가
    foodData['kcal100'] = foodData['kcal'] * (100 / foodData['food_weight'])
    foodData['protein100'] = foodData['protein'] * (100 / foodData['food_weight'])
    foodData['fat100'] = foodData['fat'] * (100 / foodData['food_weight'])
    foodData['carb100'] = foodData['carb'] * (100 / foodData['food_weight'])

    # Processing 인스턴스 생성
    processing = Processing()
    columns = ['kcal100', 'protein100', 'fat100', 'carb100']
    
    # 표준화 수행
    standardizedDf = processing.standardizeNutritionData(foodData, columns)

    # 카테고리 인코딩 (One-Hot Encoding)
    categoryEncoding = processing.oneHotEncodeCategoricalData(foodData, 'food_code_name')
    print(list(categoryEncoding.columns))

    # 카테고리 인코딩이 숫자형인지 확인하고 필요 시 변환
    categoryEncoding = categoryEncoding.apply(pd.to_numeric, errors='coerce')

    # 표준화된 영양 성분과 인코딩된 카테고리 벡터 결합
    featureVector = np.hstack((standardizedDf, categoryEncoding))
    featureColumns = ['kcal_std', 'protein_std', 'fat_std', 'carb_std'] + list(categoryEncoding.columns)
    
    # 중복 컬럼 이름 확인 및 제거
    featureColumns = pd.Index(featureColumns).unique().tolist()
    featureDf = pd.DataFrame(featureVector, columns=featureColumns)

    # numericColumns는 영양 성분과 카테고리 인코딩 컬럼으로 구성
    numericColumns = ['kcal_std', 'protein_std', 'fat_std', 'carb_std'] + list(categoryEncoding.columns)

    # 클리닝 함수 적용
    cleanedDf = cleanNumericData(featureDf[numericColumns])

    # 원본 데이터프레임에 동일한 컬럼의 데이터만 대입
    featureDf.loc[:, numericColumns] = cleanedDf


    # 사용자 프로필 및 벡터 생성
    userProfile = UserProfile(kcal=2000, protein=120, fat=44.4, carb=275, preferredCategories=['밥류', '면류', '과자류'])
    scaler = processing.getScaler()
    encoder = processing.getEncoder()

    # 최적화: 사용자가 선호하는 카테고리 컬럼만 포함
    preferredColumns = [f'food_code_name_{cat}' for cat in userProfile.preferredCategories]
    optimizedFeatureColumns = ['kcal_std', 'protein_std', 'fat_std', 'carb_std'] + preferredColumns
    optimizedFeatureDf = featureDf[optimizedFeatureColumns]

    # 사용자 벡터 생성
    userVector = createUserVector(userProfile, scaler, encoder, optimizedFeatureColumns)

    # 최적화된 콘텐츠 기반 필터링 수행
    contentRecommendation = contentBasedFiltering(userVector, foodData, optimizedFeatureDf)

    # 협업 필터링
    collaborativeRecommendation = collaborativeFiltering(user_id, foodData, feedbackData)

    # 중복 제거
    combinedRecommendation = pd.concat([contentRecommendation, collaborativeRecommendation]).drop_duplicates(subset=['food_name'])

    # 최적화를 위해 필요한 컬럼만 남김
    requiredColumns = ['food_name', 'kcal', 'protein', 'fat', 'carb']
    combinedRecommendation = combinedRecommendation[requiredColumns]

    # 목표 섭취량
    dailyTargets = {'kcal': 2000, 'protein': 120, 'fat': 44.4, 'carb': 275}
    mealTargets = {key: value / 2 for key, value in dailyTargets.items()}
    
    # 점심 세트 추천 최적화 수행
    lunchIndices = optimizeWithGeneticAlgorithm(foodData, mealTargets, min_items=3, max_items=5)
    lunchRecommendation = foodData.iloc[lunchIndices]

    # 저녁 세트 추천 최적화 수행
    dinnerIndices = optimizeWithGeneticAlgorithm(foodData, mealTargets, min_items=3, max_items=5)
    dinnerRecommendation = foodData.iloc[dinnerIndices]

    # 최종 추천 결과 출력
    print("점심 추천 세트:")
    print(lunchRecommendation[['food_name', 'kcal', 'protein', 'fat', 'carb']])

    print("\n저녁 추천 세트:")
    print(dinnerRecommendation[['food_name', 'kcal', 'protein', 'fat', 'carb']])