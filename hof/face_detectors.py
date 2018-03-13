import logging
from abc import ABCMeta, abstractproperty

import cv2
import numpy as np
import tensorflow as tf

from hof.downloader import Downloader

log = logging.getLogger(__name__)


class BaseTensorflowFaceDetector:
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

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def model_file_name(self):
        pass

    @abstractproperty
    def google_drive_doc_id(self):
        pass

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

    def draw_face(self, img, face, color):
        # bbox face
        x, y, w, h = face
        label = self.name
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y), (x + label_size[0], y + label_size[1] + base_line), color, cv2.FILLED)
        cv2.putText(img, label, (x, y + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    def download_and_extract_model(self, target_dir):
        downloader = Downloader(target_dir)
        downloader.download_file_from_google_drive(destination=self.model_file_name, id=self.google_drive_doc_id)
        downloader.extract_tar_file(destination=target_dir, zip_file_name=self.model_file_name)


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
