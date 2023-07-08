import numpy as np
import tensorflow as tf
import cv2
import visualization_utils as vis_util
from picamera2 import Picamera2, Preview
import time

def create_category_index(label_path='labelmap.txt'):
    """
    To create dictionary of label map

    Parameters
    ----------
    label_path : string, optional
        Path to labelmap.txt. The default is 'labelmap.txt'.

    Returns
    -------
    category_index : dict
        nested dictionary of labels.

    """
    f = open(label_path)
    category_index = {}
    for i, val in enumerate(f):
        if i != 0:
            val = val[:-1]
            if val != '???':
                category_index.update({(i-1): {'id': (i-1), 'name': val}})
            
    f.close()
    return category_index
def get_output_dict(image, interpreter, output_details, nms=True, iou_thresh=0.5, score_thresh=0.6):
    """
    Function to make predictions and generate dictionary of output

    Parameters
    ----------
    image : Array of uint8
        Preprocessed Image to perform prediction on
    interpreter : tensorflow.lite.python.interpreter.Interpreter
        tflite model interpreter
    input_details : list
        input details of interpreter
    output_details : list
    nms : bool, optional
        To perform non-maximum suppression or not. The default is True.
    iou_thresh : int, optional
        Intersection Over Union Threshold. The default is 0.5.
    score_thresh : int, optional
        score above predicted class is accepted. The default is 0.6.

    Returns
    -------
    output_dict : dict
        Dictionary containing bounding boxes, classes and scores.

    """
    output_dict = {
                   'detection_boxes' : interpreter.get_tensor(output_details[0]['index'])[0],
                   'detection_classes' : interpreter.get_tensor(output_details[1]['index'])[0],
                   'detection_scores' : interpreter.get_tensor(output_details[2]['index'])[0],
                   'num_detections' : interpreter.get_tensor(output_details[3]['index'])[0]
                   }

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if nms:
        output_dict = apply_nms(output_dict, iou_thresh, score_thresh)
    return output_dict

def apply_nms(output_dict, iou_thresh=0.5, score_thresh=0.6):
    """
    Function to apply non-maximum suppression on different classes

    Parameters
    ----------
    output_dict : dictionary
        dictionary containing:
            'detection_boxes' : Bounding boxes coordinates. Shape (N, 4)
            'detection_classes' : Class indices detected. Shape (N)
            'detection_scores' : Shape (N)
            'num_detections' : Total number of detections i.e. N. Shape (1)
    iou_thresh : int, optional
        Intersection Over Union threshold value. The default is 0.5.
    score_thresh : int, optional
        Score threshold value below which to ignore. The default is 0.6.

    Returns
    -------
    output_dict : dictionary
        dictionary containing only scores and IOU greater than threshold.
            'detection_boxes' : Bounding boxes coordinates. Shape (N2, 4)
            'detection_classes' : Class indices detected. Shape (N2)
            'detection_scores' : Shape (N2)
            where N2 is the number of valid predictions after those conditions.

    """
    q = 90 # no of classes
    num = int(output_dict['num_detections'])
    boxes = np.zeros([1, num, q, 4])
    scores = np.zeros([1, num, q])
    # val = [0]*q
    for i in range(num):
        # indices = np.where(classes == output_dict['detection_classes'][i])[0][0]
        boxes[0, i, output_dict['detection_classes'][i], :] = output_dict['detection_boxes'][i]
        scores[0, i, output_dict['detection_classes'][i]] = output_dict['detection_scores'][i]
    nmsd = tf.image.combined_non_max_suppression(boxes=boxes,
                                                 scores=scores,
                                                 max_output_size_per_class=num,
                                                 max_total_size=num,
                                                 iou_threshold=iou_thresh,
                                                 score_threshold=score_thresh,
                                                 pad_per_class=False,
                                                 clip_boxes=False)
    valid = nmsd.valid_detections[0].numpy()
    output_dict = {
                   'detection_boxes' : nmsd.nmsed_boxes[0].numpy()[:valid],
                   'detection_classes' : nmsd.nmsed_classes[0].numpy().astype(np.int64)[:valid],
                   'detection_scores' : nmsd.nmsed_scores[0].numpy()[:valid],
                   }
    return output_dict

def make_and_show_inference(img, interpreter, input_details, output_details, category_index, nms=True, score_thresh=0.6, iou_thresh=0.5):
    """
    Generate and draw inference on image

    Parameters
    ----------
    img : Array of uint8
        Original Image to find predictions on.
    interpreter : tensorflow.lite.python.interpreter.Interpreter
        tflite model interpreter
    input_details : list
        input details of interpreter
    output_details : list
        output details of interpreter
    category_index : dict
        dictionary of labels
    nms : bool, optional
        To perform non-maximum suppression or not. The default is True.
    score_thresh : int, optional
        score above predicted class is accepted. The default is 0.6.
    iou_thresh : int, optional
        Intersection Over Union Threshold. The default is 0.5.

    Returns
    -------
    NONE
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (300, 300), cv2.INTER_AREA)
    img_rgb = img_rgb.reshape([1, 300, 300, 3])

    interpreter.set_tensor(input_details[0]['index'], img_rgb)
    interpreter.invoke()
    
    output_dict = get_output_dict(img_rgb, interpreter, output_details, nms, iou_thresh, score_thresh)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
    img,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    min_score_thresh=score_thresh,
    line_thickness=3)
    # cv2.imshow("img", img)
    
    return output_dict

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

category_index = create_category_index()
input_shape = input_details[0]['shape']


def detect():
    img = cv2.imread("frame.jpg")
    # img = cv2.rotate(img, cv2.ROTATE_180)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows
    output_dict = make_and_show_inference(img, interpreter, input_details, output_details, category_index, score_thresh = 0.5)
    # print(output_dict)
    for i in output_dict:
        print(i, output_dict[i])
    if 10 in output_dict["detection_classes"]:
        return traffic_light(img)
    if 12 in output_dict["detection_classes"]:
        return stop_sign
    return 1

async def stop_sign():
    img = cv2.imread("frame.jpg")
    img = cv2.rotate(img, cv2.ROTATE_180)
    output_dict = mrake_and_show_inference(img, interpreter, input_details, output_details, category_index)
    if len(output_dict["detection_classes"]) == 0:
        return 1
    else:
        return 0

def traffic_light(img):

    height, width, _ = img.shape
    print("height, width: ", height, width)
    x1, y1, x2, y2 = output_dict['detection_boxes'][0]
    x1 = int(x1*height)
    x2 = int(x2*height)
    y1 = int(y1*width)
    y2 = int(y2*width)
    print("x1, y1, x2, y2: ", x1, y1, x2, y2)
    crop_img = img[x1:x2, y1:y2]
    # cv2.imshow("crop_img", crop_img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows
    return 1
    
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1640, 1232)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)
picam2.capture_file("frame.jpg")
picam2.close()

print(f"\n\n{detect()}\n\n")