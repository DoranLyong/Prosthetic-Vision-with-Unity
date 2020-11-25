
import socketserver
from queue import Queue


class MyTCPHandler(socketserver.BaseRequestHandler):

    queue  = Queue() 
    stringData = str()

    def handle(self):
       
        # 'self.request' is the TCP socket connected to the client     
        print("A client connected by: ", self.client_address[0], ":", self.client_address[1] )


        while True:
            try:
                # _server <- client 
                self.data = self.request.recv(1024).strip()   # 1024 byte for header 
                #print("Received from client: ", self.data)

                if not self.data: 
                    print("The client disconnected by: ", self.client_address[0], ":", self.client_address[1] )     
                    break                

                # _Get data from Queue stack 
                MyTCPHandler.stringData = MyTCPHandler.queue.get()     

                # _server -> client 
                #print(str(len(MyTCPHandler.stringData)).ljust(16).encode())  # <str>.ljust(16) and encode <str> to <bytearray>
                
                ###self.request.sendall(str(len(MyTCPHandler.stringData)).ljust(16).encode())  # <- Make this line ignored when you connect with C# client. 
                self.request.sendall(MyTCPHandler.stringData)  

                
                #self.request.sendall(len(MyTCPHandler.stringData).to_bytes(1024, byteorder= "big"))
                #self.request.sendall(MyTCPHandler.stringData)             

                
            except ConnectionResetError as e: 
                print("The client disconnected by: ", self.client_address[0], ":", self.client_address[1] )     
                break