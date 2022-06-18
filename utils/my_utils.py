import os
import numpy as np
import cv2
import argparse
import shutil
from collections import OrderedDict
from utils.rectification import *
from difflib import SequenceMatcher


def CRAFT_get_parser():
    result_folder = './result/'
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    os.mkdir(result_folder)

    os.mkdir("./result/CRAFT")
    os.mkdir("./result/text_recognition")

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='./data', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
                        help='pretrained refiner model')

    args = parser.parse_args()
    return args


def text_get_parser():
    parser = argparse.ArgumentParser(description='Text Recognition')
    parser.add_argument('--image_folder', required=False, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default='weights/TPS-ResNet-BiLSTM-Attn.pth',
                        help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    args = parser.parse_args()
    return args


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def identify_phase(predictions):
    max_score = 0

    """Strict checking"""

    # Main Menu
    if "auto" in predictions and "service" in predictions:
        return "Main Menu"

    # Auto Operation
    if "auto" in predictions and "operation" in predictions or "autooperation" in predictions:
        return "Auto Operation"

    # Chamber Door Control
    if "chamberdoorcontrol" in predictions or "doorinterlock" in predictions:
        return "Chamber Door Control"

    # Wiper/Elevator Position
    if "substratethickness" in predictions or ("Dose" in predictions and "Home" in predictions):
        return "Wiper/Elevator Position"

    # Log In/Out
    if "loginlout" in predictions or "loginlout" in predictions or "userloggedin" in predictions:
        return "Log In/Out"

    # Machine Status
    if "argonpressure" in predictions or "machinestatus" in predictions or \
            ("machine" in predictions and "status" in predictions):
        return "Machine Status"

    # Manual Control
    if "manualcontrol" in predictions or ("manual" in predictions and "control" in predictions):
        return "Manual Control"

    # Semi-Auto Chamber Preparation
    if "chamberpreparation" in predictions or ("duration" in predictions and "prep" in predictions) or \
            ("with" in predictions and "vacuum" in predictions) or (
            "without" in predictions and "vacuum" in predictions):
        return "Semi-Auto Chamber Preparation"

    # Service Menu
    if "servicemenu" in predictions:
        return "Service Menu"

    # Input Monitor
    if "inputmonitor" in predictions:
        return "Input Monitor"

    # Output Monitor
    if "outputmonitor" in predictions or ("prev" in predictions and "output" in predictions):
        return "Output Monitor"

    # Analogue I/O
    if "analoguelio" in predictions or "spareoutput" in predictions:
        return "Analogue I/O"

    # Servo Control Addresses
    if "plcinputs" in predictions or "plcoutputs" in predictions or "pleinputs" in predictions or \
            "pleoutputs" in predictions:
        return "Servo Control Addresses"

    # Network Addresses
    if "networkaddresses" in predictions or "pcipaddresses" in predictions:
        return "Network Addresses"

    # Laser Menu
    if "lasermenu" in predictions:
        return "Laser Menu"

    # System Tests
    if "runabovetests" in predictions or "systemitests" in predictions:
        return "System Tests"

    # RBV Settings
    if "rbvsettings" in predictions or ("fitted" in predictions or "confirm" in predictions):
        return "RBV Settings"

    # PC Power Control
    if "override" in predictions and "enabled" in predictions:
        return "PC Power Control"

    # User Settings
    if "usersettings" in predictions or ("complete" in predictions and "accepted" in predictions):
        return "User Settings"

    # Alarm/Events
    # No good targets...

    # Alarm/Event History
    # No good targets

    # Elevator Heater
    if "elevatorheater" in predictions or "settemperature" in predictions:
        return "Elevator Heater"

    """Flexible checking"""

    for prediction in predictions:
        for phase in PHASE_WORD_MAPPING.keys():
            max_score = max(max_score, similar(prediction, phase))

    if max_score < 0.4:
        return "None"

    for prediction in predictions:
        for phase in PHASE_WORD_MAPPING.keys():
            if similar(prediction, phase) == max_score:
                return phase

    # print(list(WORD_PHASE_MAPPING.keys())[scores.index(max(scores))])


def rectify_predictions(phase, predictions):
    rectified = []

    if phase not in PHASE_WORD_MAPPING.keys():
        return predictions

    for prediction in predictions:

        rectification = None
        max_score = 0

        for word in PHASE_WORD_MAPPING[phase]:
            score = similar(prediction, word)
            if score > max_score:
                max_score = score
                rectification = word

        if max_score > 0.5:
            rectified.append(rectification)
        else:
            rectified.append(prediction)

    return rectified
