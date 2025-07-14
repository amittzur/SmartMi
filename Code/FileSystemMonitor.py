import threading
import time
import os
from queue import Queue
from PIL import Image
import cv2
import numpy as np

class FileSystemMonitor:
    def __init__(self, directory, file_extensions=(".jpg", ".png", ".jpeg")):
        self.directory = directory
        self.file_extensions = file_extensions
        self.queue = Queue()
        # At startup, mark all existing folders as seen
        self.seen_folders = set(
            f for f in os.listdir(self.directory)
            if os.path.isdir(os.path.join(self.directory, f))
        )
        self.seen_files = set()

    def monitor_new_files(self):
        while True:
            try:
                # Detect new folders
                all_folders = [f for f in os.listdir(self.directory)
                               if os.path.isdir(os.path.join(self.directory, f))]
                new_folders = [f for f in all_folders if f not in self.seen_folders]
                for folder in new_folders:
                    folder_path = os.path.join(self.directory, folder)
                    # Wait before scanning the new folder to allow files to be written
                    time.sleep(5)  # Delay in seconds, adjust as needed
                    try:
                        files = [file for file in os.listdir(folder_path)
                                 if file.lower().endswith(self.file_extensions) and os.path.isfile(os.path.join(folder_path, file))]
                        for filename in files:
                            filepath = os.path.join(folder_path, filename)
                            if filepath not in self.seen_files:
                                self.queue.put(filepath)
                                self.seen_files.add(filepath)
                    except Exception as e:
                        print(f"Error reading folder {folder_path}: {e}")
                    self.seen_folders.add(folder)
            except Exception as e:
                print(f"Error monitoring directory: {e}")
            time.sleep(1)  # Adjust as needed

    def run(self, callback):
        def process_queue():
            while True:
                filepath = self.queue.get()
                try:
                    image = Image.open(filepath)
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Couldn't open image {filepath}: {e}")
                    continue
                callback(image_cv)

        monitor_thread = threading.Thread(target=self.monitor_new_files)
        monitor_thread.daemon = True
        monitor_thread.start()

        process_thread = threading.Thread(target=process_queue)
        process_thread.daemon = True
        process_thread.start()

if __name__ == '__main__':
    def my_callback(image, path):
        print(f"New image detected: {path}")

    watch_dir = r'C:\Users\gill\OneDrive - Shamir Optical Industry Ltd\Projects\SmartMi\Spark4Tester 5.9\Snapshots'  # Change as needed
    fsm = FileSystemMonitor(watch_dir)
    fsm.run(my_callback)

    while True:
        time.sleep(1)
