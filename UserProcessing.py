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
    사용자 프로필을 바탕으로 표준화된 영양소 벡터와 카테고리 벡터 생성
    """
    # 사용자 영양 성분 벡터 생성
    nutritionVector = scaler.transform([[userProfile.kcal, userProfile.protein, 
                                         userProfile.fat, userProfile.carb]]).flatten()
    
    # 사용자 선호 카테고리 반영 카테고리 벡터 생성
    categoryVector = np.zeros(encoder.categories_[0].shape[0]) # 카테고리 벡터 초기화
    for cat in userProfile.preferredCategories:
        try:
            idx = featureColumns.index(f'food_code_name_{cat}')
            categoryVector[idx - 4] = 1  # 카테고리 벡터는 영양소 4개 열 이후부터 시작
        except ValueError:
            print(f"'{cat}' 카테고리를 인코더 카테고리에서 찾을 수 없습니다.")
    
    # 사용자 최종 벡터 결합
    userVector = np.concatenate((nutritionVector, categoryVector))
    print(nutritionVector.shape)
    print(categoryVector.shape)
    return userVector