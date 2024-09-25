from flask import Flask, render_template, request, jsonify
from DataBase.DatabaseHandler import DatabaseHandler
from DataProcessing import Processing
from UserProcessing import UserProfile, createUserVector, calculateBmr
from Filtering.ContentBasedFiltering import contentBasedFiltering
from Filtering.CollaborativeFiltering import collaborativeFiltering
from GeneticAlgorithm import optimizeWithGeneticAlgorithm
import numpy as np
import pandas as pd

# Flask 앱 생성
app = Flask(__name__)

# 무한대와 NaN 값 처리 함수 정의
def cleanNumericData(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = df.clip(lower=-1e10, upper=1e10)
    return df

# DB 설정
dbConfig = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '785466',
    'database': 'food_data',
    'port': 3306,
}

# DB 연결 인스턴스 생성
dbHandler = DatabaseHandler(dbConfig)

# Flask의 메인 페이지 설정
@app.route('/')
def index():
    return render_template('foodrecommend.html')  # index.html 파일을 사용할 수 있도록 함

@app.route('/check_profile', methods=['POST'])
def check_profile():
    user_id = request.form['user_id']

    # user_profile 테이블에서 해당 user_id가 존재하는지 확인
    query = f"SELECT COUNT(*) AS profile_count FROM user_profile WHERE user_id = '{user_id}'"
    result = dbHandler.loadData(query)
    profile_exists = result.iloc[0]['profile_count'] > 0

    # 프로필 존재 여부 반환
    return jsonify({'profile_exists': profile_exists}), 200

@app.route('/save_profile', methods=['POST'])
def save_userprofile():
    # POST로 전달된 데이터 수신
    user_id = request.form['user_id']
    age = request.form['age']
    height = request.form['height']
    weight = request.form['weight']
    gender = request.form['gender']
    preferredCategories = request.form.getlist('preferredCategories')  # 다중 선택 리스트

    # food_code와 food_code_name을 그룹화
    food_codes = []
    food_code_names = []
    for food_code in preferred_categories:
        food_code = int(food_code)  # 문자열을 정수로 변환
        # 데이터베이스에서 해당 food_code_name을 가져오기
        food_code_name = dbHandler.loadData(f"SELECT food_code_name FROM food_data WHERE food_code = {food_code}").iloc[0]['food_code_name']
        food_codes.append(food_code)
        food_code_names.append(food_code_name)

    # 쉼표로 구분된 문자열로 다시 변환 (그룹화)
    food_codes_str = ','.join(map(str, food_codes))
    food_code_names_str = ','.join(food_code_names)

    # DB에 저장
    dbHandler.saveUserProfile(user_id, height, weight, age, gender, food_codes_str, food_code_names_str)

    return jsonify({"message": "Profile saved"}), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    # 사용자가 POST 요청으로 넘긴 데이터를 가져옴
    user_id = request.form['user_id']

    # food_data 및 feedback 데이터 로드
    foodData = dbHandler.loadData("SELECT * FROM food_data")
    feedbackData = dbHandler.loadData(f"SELECT user_id, food_code, food_number, rating FROM feedback WHERE user_id = '{user_id}'")
    userData = dbHandler.loadData(f"SELECT * FROM user_profile WHERE user_id = '{user_id}'")

    if feedbackData.empty:
        return jsonify({'error': '해당 사용자 ID에대한 데이터가 존재하지 않습니다.'}), 400

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
    categoryEncoding = categoryEncoding.apply(pd.to_numeric, errors='coerce')

    # 표준화된 영양 성분과 인코딩된 카테고리 벡터 결합
    featureVector = np.hstack((standardizedDf, categoryEncoding))
    featureColumns = ['kcal_std', 'protein_std', 'fat_std', 'carb_std'] + list(categoryEncoding.columns)

    # 데이터프레임 생성
    featureDf = pd.DataFrame(featureVector, columns=featureColumns)

    # 사용자 영양 섭취량 계산
    userNut = calculateBmr(userData['user_gender'].iloc[0], userData['user_weight'].iloc[0], userData['user_height'].iloc[0], userData['user_age'].iloc[0]) #*** 최적화 필요
    userProtein = (userData['user_weight'] * 1.2) / 4
    userCarb = (userNut * 0.55) / 4
    userFat = (userNut - userProtein - userCarb) / 9

    # 사용자 프로필 생성 전에 preferredCategories 값을 리스트로 변환
    preferredCategories = userData['food_code_name'].tolist()
    
    # 사용자 프로필 생성
    userProfile = UserProfile(kcal=userNut, protein=userProtein, fat=userFat, carb=userCarb, preferredCategories=preferredCategories)
    # preferredCategories 값 확인
    print("***************************************************************************************************************")
    print(f"preferredCategories: {userProfile.preferredCategories}")
    scaler = processing.getScaler()
    encoder = processing.getEncoder()

    # 최적화: 사용자가 선호하는 카테고리 컬럼만 포함
    preferredColumns = [f'food_code_name_{cat}' for cat in userProfile.preferredCategories]
    optimizedFeatureColumns = ['kcal_std', 'protein_std', 'fat_std', 'carb_std'] + preferredColumns
    optimizedFeatureDf = featureDf[optimizedFeatureColumns]

    # 사용자 벡터 생성
    userVector = createUserVector(userProfile, scaler, encoder, optimizedFeatureColumns)

    # 콘텐츠 기반 필터링
    contentRecommendation = contentBasedFiltering(userVector, foodData, optimizedFeatureDf)

    # 협업 필터링
    collaborativeRecommendation = collaborativeFiltering(user_id, foodData, feedbackData)

    # 중복 제거
    combinedRecommendation = pd.concat([contentRecommendation, collaborativeRecommendation]).drop_duplicates(subset=['food_name'])
    
    # 필요한 컬럼만 남김
    requiredColumns = ['food_name', 'kcal', 'protein', 'fat', 'carb', 'company', 'food_number', 'food_code']
    combinedRecommendation = combinedRecommendation[requiredColumns]

    # 목표 섭취량
    dailyTargets = {'kcal': userNut, 'protein': userProtein, 'fat': userFat, 'carb': userCarb}
    mealTargets = {key: value / 2 for key, value in dailyTargets.items()}

    # 점심 및 저녁 세트 추천 최적화 수행
    lunchIndices = optimizeWithGeneticAlgorithm(combinedRecommendation, mealTargets, min_items=3, max_items=5)
    dinnerIndices = optimizeWithGeneticAlgorithm(combinedRecommendation, mealTargets, min_items=3, max_items=5)

    lunchRecommendation = combinedRecommendation.iloc[lunchIndices]
    dinnerRecommendation = combinedRecommendation.iloc[dinnerIndices]

    # JSON으로 결과 반환
    return jsonify({
        'lunch': lunchRecommendation.to_dict(orient='records'),
        'dinner': dinnerRecommendation.to_dict(orient='records')
    })

# 평점 저장
@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    user_id = request.form['user_id']
    
    # 점심 평점 처리
    lunch_food_codes = request.form.getlist('lunch_food_code[]')
    lunch_food_numbers = request.form.getlist('lunch_food_number[]')
    lunch_ratings = request.form.getlist('lunch_rating[]')

    # 저녁 평점 처리
    dinner_food_codes = request.form.getlist('dinner_food_code[]')
    dinner_food_numbers = request.form.getlist('dinner_food_number[]')
    dinner_ratings = request.form.getlist('dinner_rating[]')

    # 점심과 저녁 평점 데이터를 하나의 리스트로 합침
    feedback_data = []
    
    # 점심 평점 데이터 추가
    for food_code, food_number, rating in zip(lunch_food_codes, lunch_food_numbers, lunch_ratings):
        feedback_data.append((food_code, food_number, rating))
    
    # 저녁 평점 데이터 추가
    for food_code, food_number, rating in zip(dinner_food_codes, dinner_food_numbers, dinner_ratings):
        feedback_data.append((food_code, food_number, rating))

    # 평점 데이터를 저장
    dbHandler.saveFeedback(user_id, feedback_data)

    return jsonify({'status': '평점이 저장되었습니다!'})

if __name__ == '__main__':
    app.run(debug=True)
