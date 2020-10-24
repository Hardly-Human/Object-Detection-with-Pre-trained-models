import streamlit as st
from PIL import Image
from gluoncv import model_zoo, data, utils
import matplotlib.pyplot as plt


def about():
	return ('''
			This App implements functionalities of 4 State-of-the-art Object Detection Models.
			- Single Shot Detector(SSD) Model.`	`
				[Paper](https://arxiv.org/abs/1512.02325) ` `
				[Code](https://github.com/balancap/SSD-Tensorflow)
			
			- Faster R-CNN(Region-based Convolutional Neural Networks) Model.`	`
				[Paper](https://arxiv.org/abs/1506.01497) `  `
				[Code](https://github.com/rbgirshick/py-faster-rcnn)
	
			- YOLO(You Only Look Once) Model.`	`
				[Paper](https://arxiv.org/abs/1506.02640) `  `
				[Code](https://github.com/pjreddie/darknet)
	
			- CenterNet Model.`	`
				[Paper](https://arxiv.org/abs/1904.08189) `   	`
				[Code](https://github.com/Duankaiwen/CenterNet)
			* * *
			## Why Use Pre-trained Models ?
			*Intuitively, A model is just a function that takes some 
			data as input and tries to estimate the resultant output by 
			tuning its `Weights` which is reffered to as `Training of the model`.
			This process of Training model requires a lot of Computational Capabilites.
			Especially, In`Computer Vision`tasks simple Classification and Recognition
			training tasks require GPU's to attain a adequate accuracy.*

			*Researchers, Engineers and Developers all across the globe have been working 
			to solve these problems and Thanks to `Opensource` Collaboration Communities 
			many problems are being solved with different methods, perspectives and Algorithms.*

			*Every Developer/Researcher doesn't have access to such high computation environments
			inorder to train different algorithms multiple times. Most of the problem-solvers 
			post their ideas,solutions and trained models on internet.*

			**`gluoncv` is a python package which provides us with pre-trained models 
			to solve a wide variety of Computer Vision Tasks.**

			*With packages like these any developer in world can have access to knowledge
			of using these models to build better products and **`Make world a better place`** *
			* * *
			''')

@st.cache(allow_output_mutation=True)
def load_model(model_name):
	model = model_zoo.get_model(model_name, pretrained = True)
	return model


def plot_image(model, x , img):
	class_IDs, scores, bounding_boxes = model(x)
	ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_IDs[0], 
			class_names = model.classes)
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.success("Object Detection Successful!! Plotting Image..")
	st.pyplot(plt.show())



def main():
	st.title("Object Detection App")
	st.text("Built with gluoncv and streamlit")
	st.sidebar.header("Detect objects with Pre-trained State-of-the-Art Models")

	activity = st.sidebar.selectbox("Select Activity",("About the App","Detect Objects"))
	if activity == "Detect Objects":
		image_file = st.file_uploader("Upload Image", type = ['jpg','png','jpeg'])


		img_task = st.sidebar.selectbox("Select Model",["None","SSD Model", "Faster RCNN Model","YOLO Model","CenterNet Model"])
		if img_task == "None":
			st.warning("Upload Image and Select a Task")

		elif image_file is not None:
			image1 = Image.open(image_file)
			rgb_im = image1.convert('RGB') 
			image = rgb_im.save("saved_image.jpg")
			image_path = "saved_image.jpg"

		if st.sidebar.button("Detect"):
			if img_task == "SSD Model":
				model = load_model('ssd_512_resnet50_v1_voc')
				x,img = data.transforms.presets.ssd.load_test(image_path,short =512)
				plot_image(model, x, img)
			
			elif img_task == "Faster RCNN Model":
				model = load_model('faster_rcnn_resnet50_v1b_voc')
				x,img = data.transforms.presets.rcnn.load_test(image_path,short =512)
				plot_image(model, x, img)

			elif img_task == "YOLO Model":
				model = load_model('yolo3_darknet53_voc')
				x,img = data.transforms.presets.yolo.load_test(image_path,short =512)
				plot_image(model, x, img)

			elif img_task == "CenterNet Model":
				model = load_model('center_net_resnet18_v1b_voc')
				x,img = data.transforms.presets.center_net.load_test(image_path,short =512)
				plot_image(model, x, img)





	elif activity == "About the App":
		st.subheader("About Object Detection App")
		st.markdown(about())
		st.markdown("Built with gluoncv and Streamlit by [Rehan uddin](https://hardly-human.github.io/)")
		st.success("Rehan uddin (Hardly-Human)👋😉")

if __name__ == "__main__":
	main()