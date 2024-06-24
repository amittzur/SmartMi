import threading
import sqlite3
import time
from queue import Queue
from PIL import Image
import cv2
import numpy as np
import io

class DBMonitor:
    def __init__(self, db_path):
        self.db = db_path
        self.queue = Queue()

    def monitor_new_measurements(self):
        while True:
            try:
                conn = sqlite3.connect(self.db)
                cursor = conn.cursor()

                cursor.execute('SELECT ColorTop FROM new_measurements order by id desc LIMIT 1')
                new_rows = cursor.fetchall()

                for row in new_rows:
                    self.queue.put(row)  # Send data to the main thread via queue
                    cursor.execute('DELETE FROM new_measurements')
            finally:
                conn.commit()
                conn.close()
                time.sleep(1)  # Adjust the sleep time as necessary

    def run(self, callback):
        # Function to process data from the queue
        def process_queue():
            while True:
                row = self.queue.get()
                callback(row)  # Call the provided callback with the row data

        monitor_thread = threading.Thread(target=self.monitor_new_measurements)
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
            results = cursor.fetchall()        
            return results    
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")    
        finally:
            # Close the cursor and connection
            cursor.close()
            conn.close()
            
    def get_image(self, imageID):
        results = self.execute(self.db, 'SELECT ColorTop FROM Measurement AS m JOIN MeasurementSnapshot AS ms ON m.Id == ms.MeasurementId WHERE m.PatientFirstName = ? Order By ModifiedDate DESC LIMIT 1', (imageID,))
        # Print the results
        if len(results) > 0:
            try:
                # Assuming the BLOB is in the second column, adjust the index if necessary
                image_data = results[0][0]
                # Convert the binary data to an image
                image = Image.open(io.BytesIO(image_data))
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except:
                return None
        else:
            return None

if __name__ == '__main__':
    def my_callback(raw_data):
        print("getImage")
        return raw_data

    db_path = r'C:\ProgramData\Shamir\Spark4\DB\Spark4.db'
    dm = DBMonitor(db_path)
    dm.run(my_callback)
    
    # Keep the main thread alive to continue processing
    while True:
        time.sleep(1)
