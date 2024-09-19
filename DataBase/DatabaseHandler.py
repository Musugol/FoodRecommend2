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
      REPLACE INTO feedback (user_id, food_code, food_number, rating)
      VALUES (%s, %s, %s, %s)
      """
      # 각 음식에 대해 평점 데이터를 저장
      for food_code, food_number, rating in feedback_data:
          cursor.execute(feedback_query, (user_id, food_code, food_number, rating))
      
      conn.commit()
      conn.close()