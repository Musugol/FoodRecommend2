from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def contentBasedFiltering(userVector, foodData, featureDF):
    """
    콘텐츠 기반 필터링을 사용하여 유사한 음식을 추천

    Parameters:
    userVector (np.array): 사용자 특성 벡터
    foodData(pd.DataFrame): 음식 데이터
    featureDF (pd.DataFrame): 음식의 특성 벡터 데이터프레임
    """

    # 유사도 계산
    # print("유저벡터", userVector)
    # print("featurDF", featureDF)
    similarities = cosine_similarity([userVector], featureDF.values)
    foodData['similarity'] = similarities[0]
    recommendations = foodData.sort_values(by='similarity', ascending=False)
    return recommendations