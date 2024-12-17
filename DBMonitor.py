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
    
    def monitor_state(self):
        while True:
            try:
                conn = sqlite3.connect(self.db)
                cursor = conn.cursor()

                cursor.execute('SELECT ProcessState FROM BreezeSessionSync LIMIT 1')
                new_rows = cursor.fetchall()
                for row in new_rows:
                    self.queue.put(row)  # Send data to the main thread via queue
            finally:
                conn.commit()
                conn.close()
                time.sleep(1)  # Adjust the sleep time as necessary

    def run(self, callback):
        # Function to process data from the queue
        def process_queue():
            while True:
                row = self.queue.get()
                try:
                    callback(row[0])  # Call the provided callback with the row data    
                except:
                    raise Exception("DBMonitor Couldn't get state.")

        monitor_thread = threading.Thread(target=self.monitor_state)
        monitor_thread.daemon = True  # Daemonize thread to exit with the main program
        monitor_thread.start()

        process_thread = threading.Thread(target=process_queue)
        process_thread.daemon = True
        process_thread.start()

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

    def execute_select(self, query, params=()):
        # Connect to the SQLite database
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()    
        try:
            # Execute the query
            cursor.execute(query, params)        
            # Fetch all results from the executed query
            results = cursor.fetchall()        
            return results    
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

if __name__ == '__main__':
    def my_callback(raw_data):
        print(f"state: {raw_data}")
        return raw_data
    
    # Open the config file and read the first line
    with open('config.txt', 'r') as file:
        db_dir = file.readline().strip()  # Read the first line and remove any trailing whitespace

    db_path = os.path.join(db_dir,r'SparkSync.bytes')
    print("DB Path: " + db_path)
    dm = DBMonitor(db_path)
    dm.run(my_callback)

    # Keep the main thread alive to continue processing
    while True:
        time.sleep(1)