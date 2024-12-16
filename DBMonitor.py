import threading
import sqlite3
import time
from queue import Queue
from PIL import Image
import cv2
import numpy as np
import io
import os

class DBMonitor:
    def __init__(self, db_path):
        self.db = db_path
        self.queue = Queue()
        self.initialize_FaceFeatures()
    
    def execute(self, query, params=()):
        # Connect to the SQLite database
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()    
        try:
            # Execute the query
            cursor.execute(query, params)        
            # Fetch all results from the executed query
            # Commit the transaction
            result = conn.commit()
            return result
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")    
        finally:
            # Close the cursor and connection
            cursor.close()
            conn.close()
    
    def initialize_FaceFeatures(self):
        try:
            conn = sqlite3.connect(self.db)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM FaceFeatures')
            conn.commit()
            self.execute('INSERT INTO FaceFeatures (bridge_x, bridge_y ) VALUES (?,?)', (0,0))
            conn.commit()
        finally:
            conn.close()

    def set_FaceFeatures(self, landmarks):
        results = self.execute('UPDATE FaceFeatures SET bridge_x=?, bridge_y=?', (landmarks))
        # Print the results

if __name__ == '__main__':
    def my_callback(raw_data):
        print("getImage")
        return raw_data
    
    # Open the config file and read the first line
    with open('config.txt', 'r') as file:
        db_dir = file.readline().strip()  # Read the first line and remove any trailing whitespace

    db_path = os.path.join(db_dir,r'SparkSync.bytes')
    print("DB Path: " + db_path)
    dm = DBMonitor(db_path)
    #dm.run(my_callback)
    landmarks = [1,2]
    dm.set_FaceFeatures(landmarks)
    
    # Keep the main thread alive to continue processing
    #while True:
    #    time.sleep(1)
