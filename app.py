import streamlit as st



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


def main():
  st.title("Object Detection App")
  st.text("Built with gluoncv and streamlit")
  st.sidebar.header("Detect objects with Pre-trained State-of-the-Art Models")

  activity = st.sidebar.selectbox("Select Activity",("About the App","Detect Objects"))
  if activity == "About the App":
    st.subheader("About Object Detection App")
    st.markdown(about())
    st.markdown("Built with gluoncv and Streamlit by [Rehan uddin](https://hardly-human.github.io/)")
    st.success("Rehan uddin (Hardly-Human)👋😉")

if __name__ == "__main__":
  main()