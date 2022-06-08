import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
from torch.autograd import Variable
from tqdm import tqdm

from utils.my_utils import *
from utils.craft import CRAFT
from utils import file_utils, imgproc, craft_utils


def test_net(craft_args, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, craft_args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=craft_args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if craft_args.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def CRAFT_init(craft_args):
    # CRAFT network
    net = CRAFT()
    print('Loading weights from checkpoint: ' + craft_args.trained_model)
    if craft_args.cuda:
        net.load_state_dict(copyStateDict(torch.load(craft_args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(craft_args.trained_model, map_location='cpu')))

    if craft_args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    return net


def refiner_init(craft_args):
    refine_net = None
    if craft_args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + craft_args.refiner_model + ')')
        if craft_args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(craft_args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(craft_args.refiner_model, map_location='cpu')))

        refine_net.eval()
        craft_args.poly = True

    return refine_net


def contour_detection():
    images = os.listdir("data")

    for image in tqdm(images, desc="Processing Frames"):
        frame = cv2.imread("data/" + image, flags=cv2.IMREAD_COLOR)

        _, this_frame = cv2.imencode('.jpg', frame)
        w, h, _ = frame.shape  # w = 2160, h = 3840

        # find contours
        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 150, 255, 0)
        thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find closed contours
        closed_contours = []
        for n, i in enumerate(contours):
            area, perimeter = cv2.contourArea(i), cv2.arcLength(i, True)
            if area != 0 and perimeter != 0:
                if area > perimeter:
                    if 30000 < area < 1000000:
                        epsilon = 0.01 * cv2.arcLength(i, True)
                        approx = cv2.approxPolyDP(i, epsilon, True)
                        closed_contours.append(approx)

        width_list = np.asarray([np.max(i[:, :, 0]) - np.min(i[:, :, 0]) for i in closed_contours]).reshape(-1, 1)

        # cv2.drawContours(frame, closed_contours, -1, (255, 0, 0), 10)
        # frame = cv2.resize(frame, (640, 480))
        # cv2.imshow("contour", frame)
        # cv2.waitKey(0)


def inference(craft_args):
    image_list, _, _ = file_utils.get_files(craft_args.test_folder)
    craft_result = "./result/CRAFT/"

    """Model Initialization"""
    net = CRAFT_init(craft_args)
    net.eval()
    refine_net = refiner_init(craft_args)

    """Contour Detection"""
    pass

    """Inference"""

    # load data
    for k, image_path in enumerate(tqdm(image_list, desc="Performing Inference")):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(craft_args, net, image, craft_args.text_threshold,
                                             craft_args.link_threshold, craft_args.low_text,
                                             craft_args.cuda, craft_args.poly, refine_net)

        # save score text

        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = craft_result + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=craft_result)


if __name__ == "__main__":
    craft_args = CRAFT_get_parser()
    inference(craft_args)
