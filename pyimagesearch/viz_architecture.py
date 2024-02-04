# import the necessary packages
from nn.conv import minivggnet
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import cv2

def plt_imshow(title, image):
	# convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	plt.title(title)
	plt.grid(False)
	plt.show()
	
# initialize LeNet and then write the network architecture
# visualization grpah to disk
#(pixel,channel,epoch)
model = minivggnet.MiniVGGNet.build(32, 32, 3, 10)
plot_model(model, to_file="minivgg.png", show_shapes=True)
