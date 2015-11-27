#!/usr/bin/env python

# Copyright 2015 National University of Singapore (Author: Abhinav Gupta)

from sklearn.preprocessing import MinMaxScaler
from PyQt4.phonon import Phonon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.cm as cm
import sys
import os
import shlex
import subprocess
import ctypes
import numpy as np
from OpenGL.GL import *
from PyQt4 import QtGui,QtCore
from PyQt4.QtOpenGL import *
import pyaudio
import wave
import threading

# code for vertex shader using 2D coordinates for position (part of OpenGL rendering)
vertex_code = """
    uniform float scale;
    attribute vec4 color;
    attribute vec2 position;
    varying vec4 v_color;
    void main()
    {
        gl_Position = vec4(scale*position, 0.0, 1.0);
        v_color = color;
    } """

#code for fragment shader using 4D vector for color (RGBA) (part of OpenGL rendering)
fragment_code = """
    varying vec4 v_color;
    void main()
    {
        gl_FragColor = v_color;
    } """

# no of layers of DNN to show in the GUI.
with open("layers") as f:
    number = int(f.read())

# class Window - Responsible for the User Interface of the toolkit. Acts as a means to interactively communicate with the OpenGL graphics rendering class FrWidget.

class Window(QtGui.QWidget):
    
    def __init__(self):
        super(Window, self).__init__()
         # initialize values
        self._generator = None
        self._timerId = None
        self.isrecording=False
        self.frames = 0
        
        self.uploadButton = QtGui.QPushButton("Upload Model")
        self.startrButton = QtGui.QPushButton("Start Recording")
        self.stoprButton = QtGui.QPushButton("Stop Recording")
        self.square = QtGui.QPushButton("Not Recording")
        self.square.setStyleSheet("background-color: rgb(0,255,0)")
        self.pathBox = QtGui.QLineEdit()
        self.wavpathBox = QtGui.QLineEdit()
        self.uploadwav = QtGui.QPushButton("Upload wav file")
        self.uploadwav.clicked.connect(self.selectwav)
        self.uploadButton.clicked.connect(self.upload)
        self.startrButton.clicked.connect(self.start_record)
        self.stoprButton.clicked.connect(self.stop_record)
        self.progress = QtGui.QProgressBar()

        self.topLayout = QtGui.QHBoxLayout()
        self.topLayout.addWidget(self.pathBox)
        self.topLayout.addWidget(self.uploadButton)
        self.topLayout.addWidget(self.wavpathBox)
        self.topLayout.addWidget(self.uploadwav)
        
        self.midLayout = QtGui.QHBoxLayout()
        self.midLayout.addWidget(self.square)
        self.midLayout.addWidget(self.startrButton)
        self.midLayout.addWidget(self.stoprButton)
        self.midLayout.addWidget(self.progress)

        self.mainLayout = QtGui.QGridLayout()
        self.mainLayout.addLayout(self.topLayout,0,0,1,min(number,3))
        self.mainLayout.addLayout(self.midLayout,1,0,1,min(number,3))
        
        self.setLayout(self.mainLayout)        
        self.setWindowTitle("Activation Brain")

    # Start recording the live audio sample.
    def start_record(self):
        # Clear the UI to record the audio again.
        self.square.setStyleSheet("background-color: rgb(255,0,0)")
        self.square.setText("Recording")
        if self.frames:        
            for widget in self.frSlider,self.lineedit,self.frameno,self.gobutton,self.playButton,self.pauseButton,self.stopButton, self.playsongButton:
                self.mainLayout.removeWidget(widget)
                widget.deleteLater()
                widget = None
            
            for widget in self.glWidget:
                self.mainLayout.removeWidget(widget)
                widget.deleteLater()
                widget = None

            for widget in self.label:
                self.mainLayout.removeWidget(widget)
                widget.deleteLater()
                widget = None

            self.resize(1000,100)
            self.move(300,300)
            self.progress.setValue(0)        
        
        # recording done on other thread to avoid freezing the UI.
        self.isrecording=True
        t = threading.Thread(target=self.record)
        t.start()

    # stops the recording and saves it in a wav file.
    def stop_record(self):
        self.isrecording=False
        self.square.setStyleSheet("background-color: rgb(0,255,0)")
        self.square.setText("Not Recording")
        print("* done recording")
        self.progress.setValue(30)
        stream.stop_stream()
        stream.close()
        paudio.terminate()

        wf = wave.open("output.wav", 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(paudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(recordframes))
        wf.close()
        self.createfile(os.getcwd()+"/output.wav")

    # actual record method where recording is done frame by frame.
    def record(self):
        global CHUNK
        global FORMAT
        global RATE
        CHUNK = 1024                    # buffer size to read frames, by default kept at 1024.
        FORMAT = pyaudio.paInt16        # sample format - 16 bit int
        RATE = 44100                    # record sampling rate in Hz.
        global paudio
        global stream
        global recordframes
        paudio = pyaudio.PyAudio()
        stream = paudio.open(format=FORMAT, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("* recording")
        recordframes = []
        while self.isrecording:         
            try:
                recorddata = stream.read(CHUNK)
                recordframes.append(recorddata)
            except IOError:
                pass
        return

    # select a pre-recorded wav file instead of recording live audio.
    def selectwav(self):
        wavfile2 = QtGui.QFileDialog.getOpenFileName(self, "Select file")
        self.wavpathBox.setText(wavfile2)
        self.progress.setValue(30)
        self.createfile(wavfile2)

    def playsong(self):
        if self.player.state() == Phonon.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    # Upload the path of the directory where all the files related to the model are present.
    def upload(self):
        self.foldername = QtGui.QFileDialog.getExistingDirectory(self, "Select Directory", "." , QtGui.QFileDialog.ShowDirsOnly)
        self.pathBox.setText(self.foldername)
        # open vertices file of each layer which has all the 2D coordinates of all the neurons.
        self.vertices=[[] for i in range(number)]
        for i in range(number):
            vertices = self.foldername + "/lda/" + str(i+1) +"/vertices"    
            with open(vertices) as f:
                for line in f:
                    x,y = line.split()
                    x=float(x)
                    y=float(y)
                    self.vertices[i].append((x,y))
        self.progress.setValue(10)

        global real_ver                                             
        real_ver=[[] for i in range(number)]
        
        global verlist
        verlist = [dict() for i in range(number)]
        for i in range(number):
            indices = self.foldername + "/lda/" + str(i+1) + "/indices"                        # file containing the set of vertices after Delaunay Triangulation.
            with open(indices) as f:
                for line in f:
                    p = map(int,line.split())
                    real_ver[i].append(self.vertices[i][p[0]-1])                 # store vertices in sets of 3 in order to create a triangle 
                    real_ver[i].append(self.vertices[i][p[1]-1])                 # as specified in the indices file
                    real_ver[i].append(self.vertices[i][p[2]-1])
                    
                    if p[0] in verlist[i]:
                        verlist[i][p[0]].append(p[1])
                        verlist[i][p[0]].append(p[2])
                    else:
                        verlist[i][p[0]] = [p[1]]
                        verlist[i][p[0]].append(p[2])
                    
                    if p[1] in verlist[i]:
                        verlist[i][p[1]].append(p[0])
                        verlist[i][p[1]].append(p[2])
                    else:
                        verlist[i][p[1]] = [p[0]]
                        verlist[i][p[1]].append(p[2])

                    if p[2] in verlist[i]:
                        verlist[i][p[2]].append(p[0])
                        verlist[i][p[2]].append(p[1])
                    else:
                        verlist[i][p[2]] = [p[0]]
                        verlist[i][p[2]].append(p[1])

                    verlist[i][p[0]] = list(set(verlist[i][p[0]]))
                    verlist[i][p[1]] = list(set(verlist[i][p[1]]))
                    verlist[i][p[2]] = list(set(verlist[i][p[2]]))

        self.progress.setValue(20)

    # pass the recorded audio wav file through the given DNN model in Kaldi to get the activations. 
    def createfile(self,filename):
        self.progress.setValue(40)
        wavfile = self.foldername+"/wav.scp"
        with open(wavfile,"wb") as f:
            f.write("uttid " + filename)
        command = "bash " + os.getcwd() + "/getact.sh " + self.foldername + " " + os.getcwd()
        subprocess.call(shlex.split(str(command)))              # execute the bash file to create the activations file.
        self.progress.setValue(60)
        self.initgl()

    # main method which does all the rendering and communicates with the FrWidget class.
    def initgl(self):
        # open activations file and store it in an organized manner in a list.
        with open("activations") as f:
            data = []
            i=0
            for line in f:
                temp = line.strip().split()
                if temp[-1] == "[":
                    data.append([])
                elif temp[-1] != "]":
                    ins = [float(j) for j in temp]
                    data[i].append(ins)
                else:
                    ins = [float(j) for j in temp[:-1]]
                    data[i].append(ins)
                    i+=1

        # number - number of layers in the DNN model.
        # global number
        # number = i
        self.progress.setValue(70)    
        
        for l in range(number):
            for fr in range(len(data[l])):
                maxi=max(data[l][fr])
                data[l][fr] = [i/maxi for i in data[l][fr]]
                # minmax = MinMaxScaler()
                # data[l][fr] = minmax.fit_transform(data[l][fr]).tolist()        
                
        alpha = 0.25
        vcol=[[] for i in range(number)]
        for l in range(number):
            for fr in range(len(data[l])):
                vcol[l].append([])
                for j in range(len(data[l][fr])):
                    adjver = verlist[l][j+1]
                    sum = (1-alpha)*data[l][fr][j]
                    for ver in adjver:
                        sum = sum + alpha*data[l][fr][ver-1]
                    vcol[l][fr].append(sum/(1-alpha+(alpha*len(adjver))))
        '''
        alpha = 0.5
        fcol=[[] for i in range(number)]
        for l in range(number):
            for fr in range(len(data[l])):
                fcol[l].append([])
                for j in range(len(data[l][fr])):
                    adjver = verlist[l][j+1]
                    sum = (1-alpha)*vcol[l][fr][j]
                    for ver in adjver:
                        sum = sum + alpha*vcol[l][fr][ver-1]
                    fcol[l][fr].append(sum/(1-alpha+(alpha*len(adjver))))
        '''
        self.progress.setValue(75)
        
        #jet = cm = plt.get_cmap('jet') 
        #cNorm  = colors.Normalize()
        #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        # give color to each vertex of the triangle according to the activation of that neuron
        col=[[] for i in range(number)]
        for l in range(number):
            for fr in range(len(data[l])):
                col[l].append([])
                #col[l][fr] = scalarMap.to_rgba(vcol[l][fr])
                col[l][fr] = cm.jet(plt.Normalize(min(vcol[l][fr]),max(vcol[l][fr]))(vcol[l][fr])).tolist()
                '''
                for j in vcol[l][fr]:
                    col[l][fr].append((j,0,0,1))            # can change the position of j to (0,j,0,1) for green or (0,0,j,1) for blue color gradient. 
                '''        
        
        self.progress.setValue(80)                          # Uses RGBA format. '1' enables transparency.
        self.frames = len(col[0])                   # total no of frames in the wav file.
        print "No. of frames ",self.frames


        global real_col
        real_col=[[] for i in range(number)]
        for i in range(number):
            k=0
            indices = self.foldername + "/lda/" + str(i+1) + "/indices"                        # file containing the set of vertices after Delaunay Triangulation.
            with open(indices) as f:
                for line in f:
                    p = map(int,line.split())
                    for fr in range(self.frames):
                        if k==0:
                            real_col[i].append([])
                        real_col[i][fr].append(col[i][fr][p[0]-1])          
                        real_col[i][fr].append(col[i][fr][p[1]-1])
                        real_col[i][fr].append(col[i][fr][p[2]-1])
                    k=1
                
        self.progress.setValue(90)


        self.resize(1200,600)        
        self.move(100,50)

        totaldata = []                                          
        for i in range(number):
            data = np.zeros(len(real_ver[i]), [("position", np.float32, 2), ("color", np.float32, 4)])   # data structure to store position and color for each vertex.
            data['color']    = real_col[i][0]
            data['position'] = real_ver[i]
            totaldata.append(data)                          # list of all vertices with their position and color.
        
        # add UI stuff
        self.player = Phonon.createPlayer(Phonon.MusicCategory, Phonon.MediaSource("output.wav"))
        self.playsongButton = QtGui.QPushButton("Play/Pause Audio")
        self.playsongButton.clicked.connect(self.playsong)
        self.midLayout.addWidget(self.playsongButton)
        
        self.frSlider = self.createSlider()
        self.frSlider.setValue(0)
        self.frSlider.valueChanged.connect(self.setline)

        self.frameno = QtGui.QLabel("Enter Frame No.[1-" + str(self.frames) + "]:")
        self.frameno.setFixedWidth(160)
        self.lineedit = QtGui.QLineEdit("1")
        self.lineedit.setFixedWidth(75)
        self.lineedit.returnPressed.connect(self.setslider)
        self.gobutton = QtGui.QPushButton("Go!")
        self.gobutton.setFixedWidth(75)
        self.gobutton.clicked.connect(self.setslider)

        self.playButton = QtGui.QPushButton("Play")
        self.pauseButton = QtGui.QPushButton("Pause")
        self.stopButton = QtGui.QPushButton("Stop")
        self.playButton.clicked.connect(self.playvalue)
        self.pauseButton.clicked.connect(self.pausevalue)
        self.stopButton.clicked.connect(self.stopvalue)

        self.glWidget = []
        self.label = []
        # create OpenGL windows for each layer by creating object of the FrWidget class.
        for i in range(number):
            awindow = FrWidget(i, totaldata[i])
            self.glWidget.append(awindow)
            self.frSlider.valueChanged.connect(self.glWidget[i].setFrame)
            lname = "Layer" + str(i+1)
            alabel = QtGui.QLabel(lname)
            self.label.append(alabel)
            self.mainLayout.addWidget(self.label[i],2+(i//3)*2,i%3)
            self.mainLayout.addWidget(self.glWidget[i],3+(i//3)*2,i%3)

        # add all things to mainlayout
        self.mainLayout.addWidget(self.frSlider,4+(number//3)*2,0,1,min(number,3))

        self.partLayout = QtGui.QHBoxLayout()
        self.partLayout.addWidget(self.frameno)
        self.partLayout.addWidget(self.lineedit)
        self.partLayout.addWidget(self.gobutton)
        self.partLayout.addWidget(self.playButton)
        self.partLayout.addWidget(self.pauseButton)
        self.partLayout.addWidget(self.stopButton)

        self.mainLayout.addLayout(self.partLayout,5+(number//3)*2,0,1,min(number,3))
        self.progress.setValue(100)
        
    def createSlider(self):
        slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, self.frames-1)
        slider.setSingleStep(1)
        #slider.setTickInterval(1 * 3)
        #slider.setTickPosition(QtGui.QSlider.TicksBelow)
        return slider

    # slot function for slider to change lineedit
    def setline(self, value):
    	z = value+1
    	self.lineedit.setText(str(z))

    # main function for changing activations in OpenGL windows during play
    def loopGenerator(self):
        self.val = int(self.lineedit.text())
        while self.val!=self.frames:
            self.frSlider.setValue(self.val)
            for i in range(number):
                self.glWidget[i].setFrame(self.val)
            self.val+=1
            yield

    # slot function for play button
    def playvalue(self):
        self.pausevalue()
        self._generator = self.loopGenerator()
        self._timerId = self.startTimer(0)

    # slot function for pause button
    def pausevalue(self):
        if self._timerId is not None:
            self.killTimer(self._timerId)
        self._generator = None
        self._timerId = None

    # override timerEvent function to be able to play and pause in between frames
    def timerEvent(self, event):
        if self._generator is None:
            return
        try:
            next(self._generator)
        except StopIteration:
            self.pausevalue() 
    
    # slot function for stop button to reset the slider and all the windows to frame 0
    def stopvalue(self):
        self.pausevalue()
        self.frSlider.setValue(0)
        for i in range(number):
            self.glWidget[i].setFrame(0)

    # slot function for slider
    def setslider(self):
        y = int(self.lineedit.text())
        self.frSlider.setValue(y-1)
        for i in range(number):
            self.glWidget[i].setFrame(y-1)

# class FrWidget - Responsible for OpenGL graphics rendering. Creates a window which displays activations as colours for given set of neurons.

class FrWidget(QGLWidget):

    frameChanged = QtCore.pyqtSignal(int)

    def __init__(self, no, data):
        super(FrWidget, self).__init__()
        self.no = no                            # stores the layer no.
        self.data = data                        # stores the position and color of all neurons of that particular layer
        self.tot = len(self.data['position'])

    # change the color of each window as specified by the frane
    def setFrame(self, value):
        self.frameChanged.emit(value)
        self.data['color'] = real_col[self.no][value]
        self.updateGL()

    def sizeHint(self):
        return QtCore.QSize(300, 300)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

    def initializeGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

    # actual OpenGL graphics rendering function
    def paintGL(self):
        glDrawArrays(GL_TRIANGLES, 0, self.tot)

        # Request a program and shader slots from GPU
        program  = glCreateProgram()
        vertex   = glCreateShader(GL_VERTEX_SHADER)
        fragment = glCreateShader(GL_FRAGMENT_SHADER)

        glShaderSource(vertex, vertex_code)                 # Set shaders source
        glShaderSource(fragment, fragment_code)
        glCompileShader(vertex)                             # Compile shaders
        glCompileShader(fragment)
        glAttachShader(program, vertex)                     # Attach shader objects to the program
        glAttachShader(program, fragment)
        glLinkProgram(program)                              # Build program
        glDetachShader(program, vertex)                     # Get rid of shaders (no more needed)
        glDetachShader(program, fragment)
        glUseProgram(program)                               # Make program the default program

        buffer = glGenBuffers(1)                            # Request a buffer slot from GPU
        glBindBuffer(GL_ARRAY_BUFFER, buffer)               # Make this buffer the default one
        glBufferData(GL_ARRAY_BUFFER, self.data.nbytes, self.data, GL_DYNAMIC_DRAW)

        loc = glGetUniformLocation(program, "scale")        # Bind uniforms
        glUniform1f(loc, 1.0)
        
        glBufferData(GL_ARRAY_BUFFER, self.data.nbytes, self.data, GL_DYNAMIC_DRAW)     

        # bind position from buffer to the shader
        stride = self.data.strides[0]
        offset = ctypes.c_void_p(0)
        loc = glGetAttribLocation(program, "position")
        glEnableVertexAttribArray(loc)
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        glVertexAttribPointer(loc, 2, GL_FLOAT, False, stride, offset)

        # bind color from buffer to the shader
        offset = ctypes.c_void_p(self.data.dtype["position"].itemsize)
        loc = glGetAttribLocation(program, "color")
        glEnableVertexAttribArray(loc)
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        glVertexAttribPointer(loc, 4, GL_FLOAT, False, stride, offset)


if __name__ == '__main__':
    app = QtGui.QApplication(["Activation Brain"])
    widget = Window()
    widget.resize(1000,100)
    widget.move(300,300)
    widget.show()
    app.exec_()
