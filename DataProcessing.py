from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

class Processing:
    def __init__(self):
        self.scaler = None
        self.encoder = None

    def normalizeNutritionData(data, columns):
        """
        정규화를 수행하는 함수. Min-Max Scaling을 사용하여 지정된 컬럼의 데이터를 0-1 범위로 정규화합니다.

        Parameters:
        data (pd.DataFrame): 정규화할 데이터가 포함된 데이터프레임
        columns (list): 정규화할 영양소 컬럼 리스트

        Returns:
        pd.DataFrame: 정규화된 영양소 데이터를 포함한 새로운 데이터프레임
        """
        # 선택한 컬럼 추출
        nutritionData = data[columns]

        # 무한대 값과 NaN을 처리(무한대 값을 NaN으로 대체)
        nutritionData.replace([np.inf, -np.inf], np.nan, inplace=True)

        # NaN 값을 컬럼별 평균으로 대체
        nutritionData.fillna(nutritionData.mean(), inplace=True)

        # Min-Max Scaling(정규화 수행)
        scaler = MinMaxScaler()
        normalizedData = scaler.fit_transform(nutritionData)

        # 정규화된 데이터를 새로운 데이터프레임으로 반환
        normalizedDf = pd.DataFrame(normalizedData, columns=columns)
        return normalizedDf
    
    def standardizeNutritionData(self, data, columns):
        """
        표준화를 수행하는 함수. Z-score normalization을 사용하여 지정된 컬럼의 데이터를 0-1 범위로 표준화합니다.

        Parameters:
        data (pd.DataFrame): 표준화할 데이터가 포함된 데이터프레임
        columns (list): 표준화할 영양소 컬럼 리스트

        Returns:
        pd.DataFrame: 표준화된 영양소 데이터를 포함한 새로운 데이터프레임
        """
        # 선택한 컬럼 추출
        nutritionData = data[columns]

        # 무한대 값과 NaN을 처리(무한대 값을 NaN으로 대체)
        nutritionData.replace([np.inf, -np.inf], np.nan, inplace=True)

        # NaN 값을 컬럼별 평균으로 대체
        nutritionData.fillna(nutritionData.mean(), inplace=True)

        # StandardScaler(표준화 수행)
        self.scaler = StandardScaler()
        standardizedData = self.scaler.fit_transform(nutritionData)

        # 표준화된 데이터를 새로운 데이터프레임으로 반환
        standardizedDf = pd.DataFrame(standardizedData, columns=columns)
        return standardizedDf
    
    def oneHotEncodeCategoricalData(self, data, column):
        """
        카테고리형 데이터를 원-핫 인코딩하는 함수.

        Parameters:
        data (pd.DataFrame): 원-핫 인코딩할 데이터가 포함된 데이터프레임
        column (str): 인코딩할 컬럼명

        Returns:
        pd.DataFrame: 원-핫 인코딩된 데이터를 포함한 새로운 데이터프레임
        """
        # 카테고리 컬럼이 실제 데이터프레임의 컬럼에 존재하는지 확인
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame columns: {data.columns.tolist()}")

        # 원-핫 인코딩 수행
        self.encoder = OneHotEncoder(sparse=False)
        encodedData = self.encoder.fit_transform(data[[column]])
        encodedDf = pd.DataFrame(encodedData, columns=self.encoder.get_feature_names_out([column]))
        return pd.concat([data, encodedDf], axis=1).drop(columns=[column])
    
    def getScaler(self):
        return self.scaler
    
    def getEncoder(self):
        return self.encoder