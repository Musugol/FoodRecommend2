import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class UserProfile:
    def __init__(self, kcal, protein, fat, carb, preferredCategories):
        self.kcal = kcal
        self.protein = protein
        self.fat = fat
        self.carb = carb
        self.preferredCategories = preferredCategories

def createUserVector(userProfile, scaler, encoder, featureColumns):
    """
    사용자 프로필을 바탕으로 표준화된 영양소 벡터와 사용자가 선호하는 카테고리 벡터만 포함하여 생성.
    """
    # 사용자 영양 성분 벡터 생성 전에 값을 단일 값으로 변환
    kcal = userProfile.kcal.values[0] if isinstance(userProfile.kcal, pd.Series) else userProfile.kcal
    protein = userProfile.protein.values[0] if isinstance(userProfile.protein, pd.Series) else userProfile.protein
    fat = userProfile.fat.values[0] if isinstance(userProfile.fat, pd.Series) else userProfile.fat
    carb = userProfile.carb.values[0] if isinstance(userProfile.carb, pd.Series) else userProfile.carb
    
    nutrition_data = [[kcal, protein, fat, carb]]

    print(f"Before Scaling: {nutrition_data}")

    # 사용자 영양 성분 벡터 생성
    nutritionVector = scaler.transform(nutrition_data).flatten()

    print(f"nutritionVector shape: {nutritionVector.shape}")

    # 사용자가 선호하는 카테고리 벡터 생성
    categoryVector = []
    preferredCategories = set(userProfile.preferredCategories)
    
    # featureColumns에서 카테고리 부분만 사용하여 벡터 생성
    for col in featureColumns[4:]:  # 영양소 컬럼 이후부터 카테고리 컬럼만 포함
        category_name = col.split('_')[-1]  # 'food_code_name_밥류' -> '밥류'
        if category_name in preferredCategories:
            categoryVector.append(1)
        else:
            categoryVector.append(0)

    # 최종 사용자 벡터 결합
    userVector = np.concatenate((nutritionVector, categoryVector))

    # 벡터 길이 검증
    if len(userVector) != len(featureColumns):
        print(f"유저 벡터의 길이: {len(userVector)}, 피처 컬럼 수: {len(featureColumns)}")
        print("Check the matching process between userVector and featureColumns.")
    
    print(f"userVector: {userVector}")
    return userVector

# BMR 계산 함수
def calculateBmr(gender, weight, height, age):
    """
    성별, 체중, 신장, 나이에 기반하여 BMR(기초대사량)을 계산합니다.
    """
    # 이미 gender는 문자열이므로 바로 비교
    if gender == 'male':
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    elif gender == 'female':
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

    return bmr

