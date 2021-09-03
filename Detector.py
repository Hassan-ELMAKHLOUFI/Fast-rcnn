from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo


import cv2 as cv
import numpy as np 
start_frame_number = 50



class Detector:
		def __init__(self):
			self.cfg = get_cfg()
			#load model 
			self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
			self.cfg.MODEL.WEIGHTS= model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

			self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST =0.7
			self.cfg.MODEL.DEVICE = "cpu"

			self.predictor = DefaultPredictor(self.cfg)

		def onImage(self,imagePath):
			image = cv.imread(imagePath)
			predictions =self.predictor(image)
			viz = Visualizer(image[:,:,::-1],metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
			instance_mode=ColorMode.IMAGE_BW)
			output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
			cv.imshow("Result",output.get_image()[:,:,::-1])
			cv.waitKey(0)

		def onVideo(self,videoPath):
			cap = cv.VideoCapture(videoPath)

			if(cap.isOpened()==False):
				print("Error opening the file  ...")
				return

			(sucess ,image) =cap.read()
			start_frame_number = 50
			predictions =self.predictor(image)
			while sucess:
				start_frame_number += 2
				cap.set(cv.CAP_PROP_POS_FRAMES, start_frame_number)
				predictions =self.predictor(image)
				viz = Visualizer(image[:,:,::-1],metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
				instance_mode=ColorMode.IMAGE_BW)
				output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
				cv.imshow("Result",output.get_image()[:,:,::-1])
				key = cv.waitKey(1) & 0xFF

				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break
				(sucess ,image) =cap.read()

