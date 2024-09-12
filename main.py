from DataBase.DatabaseHandler import DatabaseHandler
from DataProcessing import Processing
from UserProcessing import UserProfile, createUserVector
from Filtering.ContentBasedFiltering import contentBasedFiltering
from Filtering.CollaborativeFiltering import collaborativeFiltering
from GeneticAlgorithm import optimizeWithGeneticAlgorithm
import numpy as np
import pandas as pd

# 무한대와 NaN 값 처리
def cleanNumericData(df):
    # 무한대 값을 NaN으로 변환
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # NaN 값을 0으로 대체 (또는 적절한 값으로 대체)
    df.fillna(0, inplace=True)
    # 값 클리핑: 너무 큰 값을 적절한 범위로 제한
    df = df.clip(lower=-1e10, upper=1e10)  # 예시로 적절한 범위 설정
    return df

if __name__ == '__main__':
    # MySQL DB 연결 설정
    dbConfig = {
        'host' : '127.0.0.1',
        'user' : 'root',
        'password' : '785466',
        'database' : 'food_data',
        'port' : 3306,
    }

    # DB 연결(인스턴스 생성)
    dbHandler = DatabaseHandler(dbConfig)

    # food_data 및 feedback 데이터 로드
    foodData = dbHandler.loadData("SELECT * FROM food_data")
    ## feedback 데이터는 나중에...

    foodData['kcal100'] = foodData['kcal'] * (100/foodData['food_weight'])
    foodData['protein100'] = foodData['protein'] * (100/foodData['food_weight'])
    foodData['fat100'] = foodData['fat'] * (100/foodData['food_weight'])
    foodData['carb100'] = foodData['carb'] * (100/foodData['food_weight'])

    # Processing 인스턴스 생성
    processing = Processing()
    # 정규화(표준화)할 컬럼 지정
    columns = ['kcal100', 'protein100', 'fat100', 'carb100']
    
    # 정규화 수행
    # normalizedDf = processing.normalizeNutritionData(foodData, nutritionColumns)
    # 표준화 수행
    standardizedDf = processing.standardizeNutritionData(foodData, columns)

    # 카테고리 인코딩 (One-Hot Encoding)
    categoryEncoding = processing.oneHotEncodeCategoricalData(foodData, 'food_code_name')  # 실제 컬럼명 사용
    print(list(categoryEncoding.columns))
    # 특성 벡터 결합
    # 표준화된 영양 성분과 인코딩된 카테고리 벡터 결합
    featureVector = np.hstack((standardizedDf, categoryEncoding))

    # 결과 데이터프레임 생성 및 반환
    featureColumns = ['kcal_std', 'protein_std', 'fat_std', 'carb_std'] + list(categoryEncoding.columns)
    featureDf = pd.DataFrame(featureVector, columns=featureColumns)

    numericColumns = [ 'food_code', 'food_rep_code', 'food_nut_std', 'kcal', 'protein', 'fat', 'carb', 'natrium', 'food_weight','kcal100','protein100','fat100','carb100',
                        'food_code_name_과자류', 'food_code_name_구이류', 'food_code_name_국 및 탕류', 'food_code_name_국물류', 'food_code_name_두부류묵류', 'food_code_name_떡류',
                        'food_code_name_만두류', 'food_code_name_면류', 'food_code_name_밥류', 'food_code_name_빙과류', 'food_code_name_빵류', 'food_code_name_샌드위치류', 
                        'food_code_name_샐러드류', 'food_code_name_생채·무침류', 'food_code_name_수산가공식품류', 'food_code_name_식육가공품 및 육류', 'food_code_name_알가공품류', 
                        'food_code_name_유제품류', 'food_code_name_음료류', 'food_code_name_잼류', 'food_code_name_전·적 및 부침류', 'food_code_name_절임류조림류', 
                        'food_code_name_즉석식품류', 'food_code_name_코코아가공품초콜릿류', 'food_code_name_코코아초콜릿류', 'food_code_name_햄버거류']
    for col in numericColumns:
        featureDf[col] = pd.to_numeric(featureDf[col], errors='coerce')
    
    # 데이터 클리닝 적용: 무한대와 NaN 처리
    featureDf[numericColumns] = cleanNumericData(featureDf[numericColumns])

    # 사용자 프로필 및 벡터 생성
    userProfile = UserProfile(kcal=2000, protein=120, fat=44.4, carb=275, preferredCategories=['밥류', '면류', '과자류'])
    scaler = processing.getScaler()
    encoder = processing.getEncoder()
    userVector = createUserVector(userProfile, scaler, encoder, featureColumns)
    print("-------------------------")
    print(userVector.shape)
    print(featureDf.shape)

    # # 콘텐츠 기반 필터링
    # contentRecommendation = contentBasedFiltering(userVector, foodData, featureDf[numericColumns])
    # # 협업 필터링
    # user_id = 123 # 예시
    # collaborativeRecommendation = collaborativeFiltering(user_id, foodData, featureVector)

    # # 후보군 통합 및 최적화
    # combinedRecommendation = pd.concat([contentRecommendation, collaborativeRecommendation]).drop_duplicates()
    # dailyTargets = {'kcal': 2000, 'protein': 120, 'fat': 44.4, 'carb': 275}
    # optimizedRecommendation = optimizeWithGeneticAlgorithm(combinedRecommendation, dailyTargets)

    # # 최종 추천 결과
    # finalRecommendation = combinedRecommendation.iloc[optimizedRecommendation]
    # print(finalRecommendation[['식품명', 'kcal', 'protein', 'fat', 'carb', 'similarity']].head())