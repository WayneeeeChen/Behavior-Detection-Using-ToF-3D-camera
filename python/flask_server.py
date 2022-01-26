from threading import local
from flask import Flask, redirect, render_template, Response
from collections import deque
from flask import request
import socket
import os, sys 
#from django.shortcuts import redirect
#from django.urls import reverse
import time

app = Flask(__name__)
deq = deque(maxlen=10)
model_num = 2
frame_no = 0
tof_serial_is = 0
tof_serial = ''
Go_html = ''
Tof_SN = ''
Go_status = ''
@app.route('/')
def index():    
    global tof_serial
    #print(tof_serial)
    if (model_num == 1):
        return render_template('index.html', Go_html = 'Go', Tof_SN = tof_serial)
         #Go_html = 'Go'
         #Tof_SN = tof_serial
         #return render_template('index.html', **locals())
    elif (model_num == 2):
         return render_template('index.html', Go_html = 'Stop', Tof_SN = tof_serial)
    elif (model_num == 0):
         return render_template('index.html', Go_html = 'Initial', Tof_SN = tof_serial)
    elif (model_num == 4):
        os.system("sync")
        os.system("sudo sync")
        os.system('sudo /home/hugoliu/reboot_cmd.sh')
         #return render_template('index.html', Go_html = 'Initial')
    else:
         return render_template('index.html', Go_html = 'ExE', Tof_SN = tof_serial)
#x_reboot

#@app.route('/y_initial') 
#def get_ses(): 
    #global model_num, frame_no
 	#model_num = 0
    #frame_no = 0
    #h_name = socket.gethostname()
    #IP_address = socket.gethostbyname(h_name)
    #org_url = 'https://' + IP_address+ ':2523'
    #return redirect(org_url)

@app.route('/submtit', methods=['POST'])
def submit():
    global model_num, frame_no
    function_choise = request.values['html_fun']
    #print("*******the function******")
    print(function_choise)
    if (function_choise == 'x_initial'):
        model_num = 0
    elif (function_choise == 'x_go'):
        model_num  = 1
    elif (function_choise == 'x_stop'):
        model_num  = 2
    else:
        model_num  = 4

    frame_no = 0
    h_name = socket.gethostname()
    IP_address = socket.gethostbyname(h_name)
    org_url = 'https://' + IP_address+ ':2523'
    return redirect(org_url)

def get_frame():
    image = deq[-1]
    ret, jpeg = cv2.imencode('.jpg', image)
    return jpeg.tobytes(), jpeg.tobytes()
    #return jpeg.tobytes(), jpeg.tostring()

def gen():
    while True:
        frameBytes, _ = get_frame()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + f'{len(frameBytes)}'.encode() + b'\r\n'
               b'\r\n' + frameBytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    
    from threading import Thread
    import cv2, numpy
    
    def recvall(sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
            #print(count)
        return buf

    def num(s):
        try:
             return int(s)
        except ValueError:
             return -1

    def socket_server():
        import socket
        import ssl
        HOST = '127.0.0.1'
        PORT = 23311
        global model_num, frame_no, tof_serial, tof_serial_is
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.settimeout(100)
        server.bind((HOST, PORT))
        server.listen(10)
        #cmd = "./IPcamera &"
        #os.system(cmd)
       # print (" Listening on %s:%d " % (HOST,PORT))
        while True:
            try:
                client,addr = server.accept()
             #   client.setblocking(0)
             #   print ('Connected by ', addr)
                #i = 1
               # tof_serial = recvall(client, 128)
               # print(tof_serial)
                while True:
# head                    
#                    hugo = b''
#                    while(str(hugo) != 'hugo'):
#                         hugo = client.recv(4)
#                         client.send(hugo)
#                         print(hugo)
#                    if (hugo == 'hugo'):
#                        length = recvall(client, 20).decode('utf-8')
                   # print(frame_no)
                    length_encode = -1
                    length = b''
                    while(length_encode == -1):
                          length = client.recv(5)
                          if not length:
                              printf("No receive")
                          client.send(length)
                          length_encode = num(length.lstrip())
                  #        print("the receive len", length, len(length), length_encode)
            #        print(length_encode)
            #        print("--------frame no is : ", frame_no)
                   # imgencode = client.recv(length_encode)
                   # data = numpy.fromstring(imgencode, dtype='uint8')
                   # decimg = cv2.imdecode(data, 1)
                   # deq.append(decimg)  
                    #imgencode = b''
                    imgencode = recvall(client, length_encode)
                    #imgencode = client.recv(length_encode)
                    #print(imgencode.__len__)
                    data = numpy.fromstring(imgencode, dtype='uint8')
                    decimg = cv2.imdecode(data, 1)
                    deq.append(decimg)  #(data)
                    #event = b''
                    #event = client.recv(10)                    
                    event = recvall(client, 12)
                    #if ((frame_no < 1) &(tof_serial_is == 0)):
                    if ((frame_no < 1) & (tof_serial_is == 0)):
                        tof_serial = event
                        #print(tof_serial)
                        tof_serial_is = 1
                  #  if (event == "no run    "):
                  #       model_num = 0
                   # if (frame_no == 0):
                   #     tof_serial = recvall(client, 20)
                   #     print("the tof serial no:")
                   #     print(tof_serial)
                    frame_no = frame_no +1
                    #cv2.imshow('cam', decimg)
                    #cv2.waitKey(1)
                    #time.sleep(0.5)

                #    print("send data", model_num)
                    if (model_num == 1):
                          serverMessage = 'Go        '
                    elif (model_num == 0):
                          serverMessage = 'init      '
                    else:
                          serverMessage = 'Ack       '
                    client.send(serverMessage.encode())      
                    # status_fall = client.recv(5)
               #     print(status_fall)
                    
                    #client.sendall('Ack!')
                    #serverMessage = 'Successfully get the image.'
                    #client.sendall(serverMessage)
             #       print(serverMessage)
                    # print('====================================================')
                pass
                print("over")
                client.close()
            except:
                pass
        pass
    
    taskmgr = Thread(target=socket_server, daemon=True)
    taskmgr.start()
    #if __name__ == '__main__':
    #     app.run(host='0.0.0.0', port=80)
    app.run(host='0.0.0.0', port=2523, ssl_context='adhoc')
    #app.run()
    
                    #length = recvall(conn, 16).decode('utf-8')
                    #length = client.recv(20).decode("utf8").ljust(5)
                   #length = length.replace("x","")
                    #print(type(length_encode))
    #               print("To send message")
                   #client.send(length.encode('utf-8'))
 #               print(type(int(length)))
   
