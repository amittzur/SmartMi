
import cv2
import numpy as np
import socket
import pickle

def main():
    # Open a connection
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 9999))

    # Open the webcam
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            break

        # Convert the frame to bytes
        _, frame_bytes = cv2.imencode('.jpg', frame)

        # Send the frame size
        frame_size = len(frame_bytes)
        s.sendall(frame_size.to_bytes(4, byteorder='big'))

        # Send the frame
        s.sendall(frame_bytes.tobytes())

        # Receive the response
        response = s.recv(4096)
        # Deserialize the received data
        received_array = pickle.loads(response)
        # Process the received array
        print("Received:", received_array)

        # Display the frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    s.close()

if __name__ == "__main__":
    main()
