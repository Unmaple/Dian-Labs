import numpy as np
import torch


def compute_iou(bbox1, bbox2):

    # TODO Compute IoU of 2 bboxes.

    ...


    """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
    # computing area of each rectangles
    S_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    S_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # computing the sum_area
    sum_area = S_bbox1 + S_bbox2

    # find the each edge of intersect rectangle
    left_line = max(bbox1[1], bbox2[1])
    right_line = min(bbox1[3], bbox2[3])
    top_line = max(bbox1[0], bbox2[0])
    bottom_line = min(bbox1[2], bbox2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0
    # End of todo
