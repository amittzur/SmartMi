
import socket

# Define the IP and port to listen on
udp_ip = "127.0.0.1" # Listen on all available interfaces
udp_port = 5005
# Create the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((udp_ip, udp_port))

print(f"Listening for UDP messages on port {udp_port}...")

while True:
        data, addr = sock.recvfrom(1024) # Buffer size is 1024 bytes
        print(f"Received message from {addr}: {data.decode()}")
