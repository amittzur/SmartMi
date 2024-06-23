import threading
import sqlite3
import time
from queue import Queue

def monitor_new_measurements(db_path, queue):
    while True:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM new_measurements')
        new_rows = cursor.fetchall()

        for row in new_rows:
            queue.put(row)  # Send data to the main thread via queue
            cursor.execute('DELETE FROM new_measurements WHERE id = ?', (row[0],))

        conn.commit()
        conn.close()
        time.sleep(1)  # Adjust the sleep time as necessary

def start_monitor_thread(db_path, callback):
    queue = Queue()
    
    # Function to process data from the queue
    def process_queue():
        while True:
            row = queue.get()
            callback(row)  # Call the provided callback with the row data

    monitor_thread = threading.Thread(target=monitor_new_measurements, args=(db_path, queue))
    monitor_thread.daemon = True  # Daemonize thread to exit with the main program
    monitor_thread.start()

    process_thread = threading.Thread(target=process_queue)
    process_thread.daemon = True
    process_thread.start()

### Step 2: Define Your Callback Function

#Define a callback function that will be called with new data from the `new_measurements` table.
def my_callback(row):
    print(f'New measurement received')
    # Perform any additional actions with the row data here

### Step 3: Start the Monitor Thread

#Start the monitor thread and pass the callback function.
if __name__ == '__main__':
    db_path = r'C:\ProgramData\Shamir\Spark4\DB\Spark4.db'
    start_monitor_thread(db_path, my_callback)
    
    # Keep the main thread alive to continue processing
    while True:
        time.sleep(1)
