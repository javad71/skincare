import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def nms_pytorch(boxes, scores, labels, thresh_iou: float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location predictions for the image
            along with the class prediction scores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
        :param labels:
        :param boxes:
        :param thresh_iou:
        :param scores:
    """

    # we extract coordinates for every
    # prediction box present in P
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # we extract the confidence scores as well
    scores = scores

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []
    output_scores = []
    output_labels = []
    while len(order) > 0:

        # extract the index of the
        # prediction with the highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(boxes[idx].tolist())
        output_scores.append(scores[idx].tolist())
        output_labels.append(labels[idx].tolist())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]

        # find the IoU of every prediction in P with S
        iou = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = iou < thresh_iou
        order = order[mask]

    keep = torch.tensor(keep)
    output_scores = torch.tensor(output_scores)
    output_labels = torch.tensor(output_labels)
    return keep, output_scores, output_labels


def fast_skin_quality_and_quantity(acne_positions):
    white = 0
    black = 0
    papol = 0
    paschol = 0
    nedol = 0
    kist = 0
    grade = 0
    white_acne_grade = 2
    black_acne_grade = 4
    papol_acne_grade = 5
    paschol_acne_grade = 6
    nedol_acne_grade = 8
    kist_acne_grade = 10

    for acne in acne_positions:
        if acne == 1:
            grade = grade + white_acne_grade
            white = white + 1
        elif acne == 2:
            grade = grade + black_acne_grade
            black = black + 1
        elif acne == 3:
            grade = grade + papol_acne_grade
            papol = papol + 1
        elif acne == 4:
            grade = grade + paschol_acne_grade
            paschol = paschol + 1
        elif acne == 5:
            grade = grade + nedol_acne_grade
            nedol = nedol + 1
        else:
            grade = grade + kist_acne_grade
            kist = kist + 1

    grade = 100 - grade
    if grade < 0:
        grade = 0

    if grade >= 80:
        quality = 'خیلی خوب'
    elif 70 < grade <= 80:
        quality = 'خوب'
    elif 60 < grade <= 70:
        quality = 'متوسط'
    elif 50 < grade <= 60:
        quality = 'بد'
    else:
        quality = 'خیلی بد'

    return grade, white, black, papol, paschol, nedol, kist, quality


