from styx_msgs.msg import TrafficLight

from keras.preprocessing import image
from keras.models import load_model

import numpy as np

import rospy

class TLClassifier(object):
    def __init__(self):

        #Todo: move to config
        dir = '/capstone/ros/src/tl_detector/light_classification/'
        self.loaded_model = load_model(dir + 'traffic_light_classifier.h5')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        img = image.img_to_array(image)
        img = np.expand_dims(img, axis=0)

        prediction = self.loaded_model.predict(img)

        light = TrafficLight.UNKNOWN
        score_treshold = 0.8
        if prediction[0] > score_treshold:
            light = TrafficLight.GREEN
        if prediction[1] > score_treshold:
            light = TrafficLight.YELLOW
        if prediction[2] > score_treshold:
            light = TrafficLight.RED

        rospy.logerr("Light predicted: {}".format(light))

        return light
