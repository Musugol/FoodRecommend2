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
    # 사용자 영양 성분 벡터 생성
    nutritionVector = scaler.transform([[userProfile.kcal, userProfile.protein, 
                                         userProfile.fat, userProfile.carb]]).flatten()
    
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