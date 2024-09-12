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

  def saveFeedback(self, user_id, food_id, rating):
    print('start saveFeedback')
    conn = self.connectDB()
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO feedback (user_id, food_id, rating) VALUES (%s, %s, %s), (user_id, food_id, rating)")

    conn.commit()
    conn.close()
    print('end saveFeedback')