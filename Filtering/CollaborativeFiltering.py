import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def collaborativeFiltering(user_id, foodData, feedback_data):
    """
    협업 필터링을 사용하여 음식을 추천

    Parameters:
    user_id(int): 사용자 ID
    feedback_data (pd.Dataframe): 사용자 피드백 데이터 (평점 등)
    foodData(pd.DataFrmae): 음식 데이터

    Returns:
    pd.DataFrame: 협업 필터링으로 추천된 음식 리스트
    """

    # 사용자-아이템 행렬 생성
    userItemMatrix = feedback_data.pivot(index='user_id', columns='food_id', values='rating').fillna(0)
    # SVD 모델 적용
    svd = TruncatedSVD(n_components=20)
    matrixSvd = svd.fit_transform(userItemMatrix)

    # 사용자 유사도 계산
    userIndex = userItemMatrix.index.get_loc(user_id)
    userSimilarity = cosine_similarity([matrixSvd[userIndex]], matrixSvd)[0]
    
    # 유사한 사용자들의 평균 평점 기반으로 음식 추천
    similarUsers = userItemMatrix.index[userSimilarity > 0.5]
    recommendedFoods = foodData[foodData['food_id'].isin(userItemMatrix.loc[similarUsers].mean(axis=0).nlargest(10).index)]

    return recommendedFoods