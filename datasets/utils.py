import cv2
import numpy as np
import torch
import torchvision

to_tensor = torchvision.transforms.ToTensor()
draw_confidence_threshold = 0.65


def align(image, eyesCenters, desiredLeftEye, desiredFaceWidth):

    # compute the center of mass for each eye
    leftEyeCenter, rightEyeCenter = eyesCenters

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = desiredRightEyeX - desiredLeftEye[0]
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = (
        (leftEyeCenter[0] + rightEyeCenter[0]) // 2,
        (leftEyeCenter[1] + rightEyeCenter[1]) // 2,
    )

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceWidth * desiredLeftEye[1]
    M[0, 2] += tX - eyesCenter[0]
    M[1, 2] += tY - eyesCenter[1]

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceWidth)
    output = cv2.warpAffine(np.array(image), M, (w, h), flags=cv2.INTER_CUBIC)

    # return the aligned face
    return output


@torch.no_grad()
def preprocess_image(model, filter_labels, device, transform, pil_image):

    image = to_tensor(pil_image).to(device)

    # print("Running image through model... ", flush=True)
    outputs = model([image])[0]  # We index 0 because we are using batch size 1

    scores = outputs["scores"]
    top_scores_filter = scores > draw_confidence_threshold
    top_boxes = outputs["boxes"][top_scores_filter]
    top_labels = outputs["labels"][top_scores_filter]

    left_eye_label = {14: "eye-dl-l", 16: "eye-dr-l", 18: "eye-fl"}
    right_eye_label = {15: "eye-dl-r", 17: "eye-dr-r", 19: "eye-fr"}

    keep_image = False
    eyel = None
    eyer = None

    for idx, label_t in enumerate(top_labels):

        label = int(label_t)
        if label in filter_labels:
            keep_image = True
        if label in left_eye_label.keys():
            eyel = top_boxes[idx].cpu()
        if label in right_eye_label.keys():
            eyer = top_boxes[idx].cpu()

        if eyel is not None and eyer is not None and keep_image:
            break

    if transform == 1:
        # zoomed crop
        eyel_pos = (0.25, 0.4)
    elif transform == 2:
        # All face crop
        eyel_pos = (0.35, 0.4)

    if eyel is not None and eyer is not None and keep_image:
        return (
            align(
                pil_image,
                (
                    ((eyel[0] + eyel[2]) / 2, (eyel[1] + eyel[3]) / 2),
                    ((eyer[0] + eyer[2]) / 2, (eyer[1] + eyer[3]) / 2),
                ),
                eyel_pos,
                224,
            )
            if transform
            else np.array(pil_image)
        )

    else:
        return None
