import pandas as pd
import pymysql

class DatabaseHandler:
  def __init__(self, config):
    self.config = config

  def connectDB(self):
    return pymysql.connect(**self.config)

  def loadData(self, query):
    conn = self.connectDB()
    data = pd.read_sql(query, conn)
    conn.close()
    return data

  def saveFeedback(self, user_id, feedback_data):
      """
      평점 데이터를 저장하는 메소드
      feedback_data는 각 평점에 대해 (food_code, food_number, rating)의 리스트 형태로 전달됨
      """
      conn = self.connectDB()
      cursor = conn.cursor()
      feedback_query = """
      INSERT INTO feedback (user_id, food_code, food_number, rating)
      VALUES (%s, %s, %s, %s)
      """
      # 각 음식에 대해 평점 데이터를 저장
      for food_code, food_number, rating in feedback_data:
          cursor.execute(feedback_query, (user_id, food_code, food_number, rating))
      
      conn.commit()
      conn.close()

  def saveUserProfile(self, user_id, user_height, user_weight, user_age, user_gender, food_codes, food_code_names):
      """
      사용자 프로필 저장 메소드.
      food_codes 및 food_code_names는 쉼표로 구분된 문자열로 전달됩니다.
      """
      conn = self.connectDB()
      cursor = conn.cursor()

      # SQL 쿼리 작성
      query = """
      INSERT INTO user_profile (user_id, user_height, user_weight, user_age, user_gender, food_code, food_code_name)
      VALUES (%s, %s, %s, %s, %s, %s, %s)
      """
      values = (user_id, user_height, user_weight, user_age, user_gender, food_codes, food_code_names)

      # 쿼리 실행 및 데이터베이스에 저장
      try:
          cursor.execute(query, values)
          conn.commit()
      except Exception as e:
          conn.rollback()
          raise e
      finally:
          cursor.close()
          conn.close()