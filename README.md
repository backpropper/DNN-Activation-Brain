# DNN-Activation-Brain

I have developed a toolkit which visualizes the hidden layer activations of a deep neural network. I used OpenGL shader libraries in python for rendering graphics. The toolkit allows loading different models for visualization. It has a functionality of recording live audio samples which are subsequently feedforwarded through a deep neural network in Kaldi to get the activations.
Work is constantly been updated.

Libraries required (for Linux):
	*Run the following commands on Debian/Ubuntu systems (Package name same in other distros):
	- sudo apt-get install python python-numpy python-scipy python-matplotlib python-pip python-opengl python-qt4*
	- python pip -m install pyaudio
	*For installing Kaldi:
	- First cd into the directory where you have downloaded current repository.
	- git clone https://github.com/abhinavgupta94/kaldi.git kaldi
	- cd kaldi/tools/; make; cd ../src; ./configure; make


Files explained:
- slider.py: Main python application file
- lda - folder conatining "vertices" and "indices" file for each of the layers
- vertices: a 2048 x 2 matrix (each row is a 2D coordinate for each hidden unit computed using t-SNE plots)
- indices: a 4069 x 3 matrix (each row is a tuple containing indices to the vertices that form a triangle computed after Delaunay Triangulation)
- getact.sh: bash file to get activations from the Kaldi model. Creates the activations file. Need to change the Kaldi installation path at the top of the file.
- activities & activities.realtime: sample activations of previously recorded wav files (can be used instead of recording live audio or if Kaldi is not installed)

How to run:
- Run python slider.py.
- After the window opens, select Upload Model and choose the directory where the folder containing the "vertices" and "indices" of all the layers are present files are present along with the other DNN model files (here it is the "model" directory). 
- Then select start recording to start recording the live audio sample. Select stop recording to stop it and see the magic.
