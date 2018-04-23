import os
import logging
from abc import ABCMeta, abstractmethod, abstractproperty

import cv2
import numpy as np
import tensorflow as tf

from hof.downloader import Downloader

log = logging.getLogger(__name__)


class FaceDetector:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def detect(self, image, include_score=False, draw_faces=True, color=(0, 255, 0), min_confidence=None):
        pass

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def model_file_name(self):
        pass

    @abstractproperty
    def google_drive_doc_id(self):
        pass

    @staticmethod
    def _ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def download_and_extract_model(self, target_dir):
        self._ensure_dir(target_dir)
        downloader = Downloader(target_dir)
        downloader.download_file_from_google_drive(destination=self.model_file_name, id=self.google_drive_doc_id)
        downloader.extract_tar_file(destination=target_dir, zip_file_name=self.model_file_name)

    def draw_face(self, img, face, color):
        # bbox face
        x, y, w, h = face
        label = self.name
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y), (x + label_size[0], y + label_size[1] + base_line), color, cv2.FILLED)
        cv2.putText(img, label, (x, y + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


class BaseTensorflowFaceDetector(FaceDetector):
    __metaclass__ = ABCMeta

    def __init__(self, min_confidence, checkpoint):
        super(BaseTensorflowFaceDetector, self).__init__()
        self.min_confidence = min_confidence
        self.detection_graph = tf.Graph()
        self.checkpoint = checkpoint

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            if not tf.gfile.Exists(self.checkpoint):
                log.info('Checkpoint not found. Triggering download.')
                self.download_and_extract_model('models/')

            with tf.gfile.GFile(self.checkpoint, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph)

    def detect(self, image, include_score=True, draw_faces=True, color=(0, 255, 0), min_confidence=None):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        faces = []
        im_height, im_width, _ = image.shape

        min_confidence = min_confidence if min_confidence is not None else self.min_confidence

        for i in range(boxes.shape[0]):
            if scores[i] >= self.min_confidence:
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)

                x, y, w, h = int(left), int(top), int(right - left), int(bottom - top)
                if include_score:
                    faces.append([x, y, w, h, scores[i]])
                else:
                    faces.append([x, y, w, h])

                if draw_faces:
                    self.draw_face(image, (x, y, w, h), color)

        return faces


class RfcnResnet101FaceDetector(BaseTensorflowFaceDetector):

    @property
    def name(self):
        return 'RFCN'

    @property
    def model_file_name(self):
        return 'rfcn_resnet101_widerface_91674.tar.gz'

    @property
    def google_drive_doc_id(self):
        return '1is7Ldv9ASYNcrv2GyXS7EaV58UaqhuFQ'

    def __init__(self, min_confidence, checkpoint='models/rfcn_resnet101_91674/frozen_inference_graph.pb'):
        super(RfcnResnet101FaceDetector, self).__init__(min_confidence, checkpoint=checkpoint)


class FasterRCNNFaceDetector(BaseTensorflowFaceDetector):

    @property
    def name(self):
        return 'fasterRCNN'

    @property
    def model_file_name(self):
        return 'faster_rcnn_inception_resnet_v2_atrous_widerface_65705.tar.gz'

    @property
    def google_drive_doc_id(self):
        return '1bMdKHMcVidrG7BUvoIk6cCcEGKhBFvcc'

    def __init__(self, min_confidence,
                 checkpoint='models/faster_rcnn_inception_resnet_v2_atrous_65705/frozen_inference_graph.pb'):
        super(FasterRCNNFaceDetector, self).__init__(min_confidence, checkpoint=checkpoint)


class SSDMobileNetV1FaceDetector(BaseTensorflowFaceDetector):

    @property
    def name(self):
        return 'ssd'

    @property
    def model_file_name(self):
        return 'ssd_mobilenet_v1_widerface_106650.tar.gz'

    @property
    def google_drive_doc_id(self):
        return '1NT3PLBHa4cYj_RmKlRrCZSWMKMct2-26'

    def __init__(self, min_confidence, checkpoint='models/ssd_mobilenet_v1_106650/frozen_inference_graph.pb'):
        super(SSDMobileNetV1FaceDetector, self).__init__(min_confidence, checkpoint=checkpoint)


class YOLOv2FaceDetector(FaceDetector):
    inWidth = 416
    inHeight = 416
    inScaleFactor = 1 / float(255)

    @property
    def name(self):
        return 'YOLOv2'

    @property
    def model_file_name(self):
        return 'yolo-widerface-v2.tar.gz'

    @property
    def google_drive_doc_id(self):
        return '1xm1zLd4Up6-lBBKegGSD1PCjsN2DYhX9'

    def cfg(self):
        return '{}/yolo-widerface.cfg'.format(self.model_dir())

    def model(self):
        return '{}/yolo-widerface_final.weights'.format(self.model_dir())

    def model_dir(self):
        return 'models/{}/'.format(self.name)

    def __init__(self, min_confidence):
        super(YOLOv2FaceDetector, self).__init__()
        self.min_confidence = min_confidence
        self.cfg = self.cfg()
        self.model = self.model()
        self.names = 'models/yolo/obj.names'

        if not os.path.exists(self.cfg):
            log.info('Cfg not found. Triggering download.')
            self.download_and_extract_model(self.model_dir())

        self.net = cv2.dnn.readNetFromDarknet(self.cfg, self.model)
        if self.net.empty():
            raise ValueError('Could not load net')

    def detect(self, image, include_score=False, draw_faces=True, color=(0, 255, 0), min_confidence=None):
        blob = cv2.dnn.blobFromImage(image, self.inScaleFactor, (self.inWidth, self.inHeight), (0, 0, 0), True, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        rows, cols, _ = image.shape
        faces = []
        for i in range(detections.shape[0]):
            confidence = detections[i, 5]
            if confidence > self.min_confidence:
                x_center = detections[i, 0] * cols
                y_center = detections[i, 1] * rows
                width = detections[i, 2] * cols
                height = detections[i, 3] * rows
                xmin = int(round(x_center - width / 2))
                ymin = int(round(y_center - height / 2))
                if include_score:
                    faces.append([xmin, ymin, int(width), int(height), confidence])
                else:
                    faces.append([xmin, ymin, int(width), int(height)])

                if draw_faces:
                    self.draw_face(image, (xmin, ymin, int(width), int(height)), color)
        return faces


class TinyYOLOFaceDetector(YOLOv2FaceDetector):

    def __init__(self, min_confidence):
        super(TinyYOLOFaceDetector, self).__init__(min_confidence)

    @property
    def name(self):
        return 'TinyYOLO'

    @property
    def model_file_name(self):
        return 'tiny-yolo-widerface-v2.tar.gz'

    @property
    def google_drive_doc_id(self):
        return '1wAy6-XWDBHMMawgPfKv2hp_Q5NBYqK1n'

    def cfg(self):
        return '{}/tiny-yolo-widerface.cfg'.format(self.model_dir())

    def model(self):
        return '{}/tiny-yolo-widerface_final.weights'.format(self.model_dir())


class ViolaJonesFaceDetector(FaceDetector):

    @property
    def model_file_name(self):
        pass

    @property
    def google_drive_doc_id(self):
        pass

    @property
    def name(self):
        return 'ViolaJones'

    def __init__(self):
        super(ViolaJonesFaceDetector, self).__init__()
        self.scale_factor = 1.1
        self.min_neighbors = 3
        self.min_size = (30, 30)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame, include_score=False, draw_faces=True, color=(0, 255, 0), min_confidence=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
                                                   minSize=self.min_size, flags=cv2.CASCADE_SCALE_IMAGE)

        if draw_faces:
            for face in faces:
                self.draw_face(frame, face, color)

        return faces
