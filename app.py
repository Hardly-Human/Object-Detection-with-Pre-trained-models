import streamlit as st
from PIL import Image
from gluoncv import model_zoo, data, utils
import matplotlib.pyplot as plt
import os

def about():
    return ('''
        This App implements functionalities of 4 State-of-the-art Object Detection Models.
        - Single Shot Detector(SSD) Model.` `
        [Paper](https://arxiv.org/abs/1512.02325) ` `
        [Code](https://github.com/balancap/SSD-Tensorflow)

        - Faster R-CNN(Region-based Convolutional Neural Networks) Model.`  `
        [Paper](https://arxiv.org/abs/1506.01497) `  `
        [Code](https://github.com/rbgirshick/py-faster-rcnn)

        - YOLO(You Only Look Once) Model.`  `
        [Paper](https://arxiv.org/abs/1506.02640) `  `
        [Code](https://github.com/pjreddie/darknet)

        - CenterNet Model.` `
        [Paper](https://arxiv.org/abs/1904.08189) `     `
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

def about_ssd_model():
    return ("""
        # About Single Shot Detection(SSD) Models 
        ### [Paper](https://arxiv.org/abs/1512.02325) `  ` [Code](https://github.com/balancap/SSD-Tensorflow)

        SSD is designed for object detection in real-time. 
        Single Shot detector like YOLO takes only one shot 
        to detect multiple objects present in an image using multibox.
        It is significantly faster in speed and high-accuracy object 
        detection algorithm. 

        High speed and accuracy of SSD using relatively low resolution 
        images is attributed due to following reasons
        - Eliminates bounding box proposals like the ones used in RCNN’s.
        - Includes a progressively decreasing convolutional filter for 
        predicting object categories and offsets in bounding box locations.

        High detection accuracy in SSD is achieved by using multiple boxes 
        or filters with different sizes, and aspect ratio for object detection. 
        It also applies these filters to multiple feature maps from the later 
        stages of a network. This helps perform detection at multiple scales.

        ![SSD Model Architecture](https://miro.medium.com/max/1000/1*hdSE1UCV7gA7jzfQ03EnWw.png)        

        **Source : [SSD : Single Shot Detector for object detection using MultiBox](https://towardsdatascience.com/ssd-single-shot-detector-for-object-detection-using-multibox-1818603644ca)** 
        """)

def about_faster_RCNN_model():
    return ("""
        # About Faster R-CNN(Region-based Convolutional Neural Networks) Models 
        ### [Paper](https://arxiv.org/abs/1506.01497) `  ` [Code](https://github.com/rbgirshick/py-faster-rcnn)

        In the R-CNN family of papers, the evolution between versions was usually in terms 
        of computational efficiency (integrating the different training stages), reduction 
        in test time, and improvement in performance (mAP). These networks usually consist of

        - A region proposal algorithm to generate “bounding boxes” or locations of possible objects in the image. 
        - A feature generation stage to obtain features of these objects, usually using a CNN.
        - A classification layer to predict which class this object belongs to
        - A regression layer to make the coordinates of the object bounding box more precise.

        The only stand-alone portion of the network left in Fast R-CNN was the region proposal algorithm. 
        Both R-CNN and Fast R-CNN use CPU based region proposal algorithms, 
        Eg- the Selective search algorithm which takes around 2 seconds per image and runs on CPU computation. 
        The Faster R-CNN paper fixes this by using another convolutional network (the RPN) to generate the region proposals. 
        This not only brings down the region proposal time from 2s to 10ms per image but also allows the region proposal 
        stage to share layers with the following detection stages, causing an overall improvement in feature representation. 
        In the rest of the article, “Faster R-CNN” usually refers to a detection pipeline that uses the RPN as a region proposal 
        algorithm, and Fast R-CNN as a detector network.
        ![Faster RCNN Model Architecture](https://miro.medium.com/max/700/1*S_-8lv4zP3W8IVfGP6_MHw.jpeg)        

        **Source : [Faster R-CNN for object detection](https://towardsdatascience.com/faster-r-cnn-for-object-detection-a-technical-summary-474c5b857b46)** 

        """)


def about_yolo_model():
    return ("""
        # About YOLO(You Only Look Once) Models 
        ### [Paper](https://arxiv.org/abs/1506.02640) `  ` [Code](https://github.com/pjreddie/darknet)

        Compared to other region proposal classification networks (fast RCNN) which perform detection on various region 
        proposals and thus end up performing prediction multiple times for various regions in a image, Yolo architecture 
        is more like FCNN (fully convolutional neural network) and passes the image (nxn) once through the FCNN and output 
        is (mxm) prediction. This the architecture is splitting the input image in mxm grid and for each grid generation 2 
        bounding boxes and class probabilities for those bounding boxes. Note that bounding box is more likely to be larger 
        than the grid itself.

        A single convolutional network simultaneously predicts multiple bounding boxes and class probabilities for those boxes. 
        YOLO trains on full images and directly optimizes detection performance. This unified model has several benefits over 
        traditional methods of object detection. First, YOLO is extremely fast. Since we frame detection as a regression problem 
        we don’t need a complex pipeline. We simply run our neural network on a new image at test time to predict detections. 
        Our base network runs at 45 frames per second with no batch processing on a Titan X GPU and a fast version runs at more 
        than 150 fps. This means we can process streaming video in real-time with less than 25 milliseconds of latency.

        Second, YOLO reasons globally about the image when making predictions. Unlike sliding window and region proposal-based 
        techniques, YOLO sees the entire image during training and test time so it implicitly encodes contextual information 
        about classes as well as their appearance. Fast R-CNN, a top detection method, mistakes background patches in an image 
        for objects because it can’t see the larger context. YOLO makes less than half the number of background errors compared 
        to Fast R-CNN.

        Third, YOLO learns generalizable representations of objects. When trained on natural images and tested on artwork, 
        YOLO outperforms top detection methods like DPM and R-CNN by a wide margin. Since YOLO is highly generalizable it is 
        less likely to break down when applied to new domains or unexpected inputs.

        Our network uses features from the entire image to predict each bounding box. It also predicts all bounding boxes 
        across all classes for an image simultaneously. This means our network reasons globally about the full image and 
        all the objects in the image. The YOLO design enables end-to-end training and realtime speeds while maintaining high 
        average precision.

        ![Yolo Model Architecture](https://miro.medium.com/max/700/1*m8p5lhWdFDdapEFa2zUtIA.jpeg)        

        ![Yolo Model Architecture](https://miro.medium.com/max/700/1*ZbmrsQJW-Lp72C5KoTnzUg.jpeg)        

        **Source : [YOLO — You only look once, real time object detection explained](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006)** 

        """)

def about_centernet_model():
    return ("""
        # About CenterNet Models 
        ### [Paper](https://arxiv.org/abs/1904.08189) `     ` [Code](https://github.com/Duankaiwen/CenterNet)

        The CenterNet paper is a follow-up to the CornerNet. The CornerNet uses a pair of corner key-points to overcome the 
        drawbacks of using anchor-based methods. However, the performance of the CornerNet is still restricted when detecting 
        the boundary of the objects since it has a weak ability referring to the global information of the object. The authors 
        of the CenterNet paper analyzes the performance of the CornerNet. They found out that the false discovery rate of 
        CornerNet on the MS-COCO validation set is high(especially on small objects) due to the proportion of the incorrect 
        bounding boxes.

        CenterNet tries to overcome the restrictions encountered in CornerNet. As it’s mentioned in the name, 
        the network uses additional information (centeredness information) to perceive the visual patterns within 
        each proposed region. Now instead of using two corner information, it uses triplets to localize objects. 
        The work states that if a predicted bounding box has a high IoU with the ground-truth box, then the probability 
        that the center keypoint in its central region is predicted as the same class is high, and vice versa. 
        During inference, given the corner points as the proposals, the network verifies whether the corner proposal 
        is indeed an object by checking if there’s a center key point of the same class falling within its central region. 
        The additional use of object centeredness keeps the network as one stage detector but inherits the functionality 
        of RoI polling like it’s used in two-stage detectors.

        ![CenterNet Model Architecture](https://miro.medium.com/max/700/1*rB7m0USZEUAF7gqcHayoEQ.png)        

        **Source : [CenterNet Keypoint Triplets for Object Detection](https://towardsdatascience.com/centernet-keypoint-triplets-for-object-detection-review-a314a8e4d4b0)** 

        """)


@st.cache(allow_output_mutation=True)
def load_model(model_name):
    model = model_zoo.get_model(model_name, pretrained = True)
    return model


def plot_image(model, x , img):
    st.subheader("Processed Image :")
    class_IDs, scores, bounding_boxes = model(x)
    ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_IDs[0], 
      class_names = model.classes)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.success("Object Detection Successful!! Plotting Image..")
    st.pyplot(plt.show())


###############################################################
# Deleting User's Data...........................
###############################################################
def delete_image(img_path):
    os.remove(img_path)

#############################################################
# main
#############################################################

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
        st.image(image1,width = 700,height = 600)
    else:
        pass


    if st.sidebar.button("Detect"):
        if img_task == "SSD Model":
            model = load_model('ssd_512_resnet50_v1_voc')
            x,img = data.transforms.presets.ssd.load_test(image_path,short =512)
            plot_image(model, x, img)
            delete_image(image_path)

        elif img_task == "Faster RCNN Model":
            model = load_model('faster_rcnn_resnet50_v1b_voc')
            x,img = data.transforms.presets.rcnn.load_test(image_path,short =512)
            plot_image(model, x, img)
            delete_image(image_path)

        elif img_task == "YOLO Model":
            model = load_model('yolo3_darknet53_voc')
            x,img = data.transforms.presets.yolo.load_test(image_path,short =512)
            plot_image(model, x, img)
            delete_image(image_path)

        elif img_task == "CenterNet Model":
            model = load_model('center_net_resnet18_v1b_voc')
            x,img = data.transforms.presets.center_net.load_test(image_path,short =512)
            plot_image(model, x, img)
            delete_image(image_path)

    # About Models...
    if img_task == "SSD Model":
        # About SSD Model
        st.sidebar.header("Learn more about SSD Models") 
        if st.sidebar.checkbox("Show about SSD Models"):
            st.write(about_ssd_model())
        else:
            pass
    elif img_task == "Faster RCNN Model":
        st.sidebar.header("Learn more about Faster RCNN Models") 
        if st.sidebar.checkbox("Show about Faster RCNN Models"):
            st.write(about_faster_RCNN_model())
        else:
            pass
    elif img_task == "YOLO Model":
        st.sidebar.header("Learn more about YOLO Models") 
        if st.sidebar.checkbox("Show about YOLO Models"):
            st.write(about_yolo_model())
        else:
            pass
    elif img_task == "CenterNet Model":
        st.sidebar.header("Learn more about CenterNet Models") 
        if st.sidebar.checkbox("Show about CenterNet Models"):
            st.write(about_centernet_model())
        else:
            pass

    elif activity == "About the App":
        st.subheader("About Object Detection App")
        st.markdown(about())
        st.markdown("Built with gluoncv and Streamlit by [Rehan uddin](https://hardly-human.github.io/)")
        st.success("Rehan uddin (Hardly-Human)👋😉")
        st.markdown("### [Give Feedback](https://www.iamrehan.me/forms/feedback_form/feedback_form.html)\
        `            `[Report an Issue](https://www.iamrehan.me/forms/report_issue/report_issue.html)")
        

if __name__ == "__main__":
    main()
