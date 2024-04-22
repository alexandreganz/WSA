# imports
import cv2
import numpy as np


def read_video(path):
    # load video to frames
    frames = []
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"--- Reading video ---")
    print(f"video path: {path}")
    print(f"fps: {fps}")
    while cap.isOpened():
        ret, frame = cap.read()

        # ret is True if frame is read correctly, so at end of video ret is False
        if not ret:
            break

        frames.append(frame)
    cap.release()
    print(f"number of frames: {len(frames)}")

    return frames, fps


def write_video(path, frames, fps):
    print("--- Writing video ---")
    print(f"video path: {path}")
    print(f"fps: {fps}")
    h, w, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_video = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        output_video.write(frame)
    output_video.release()
    print(f"number of frames: {len(frames)}")


def determine_area_to_mask(image):
    print("draw a rectangle around the area you want to mask by pressing the left mouse button and dragging the mouse")
    print("press 'q' to exit and save")
    drawing = False  # true if mouse is pressed
    ix, iy = -1, -1

    # mouse callback function
    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, drawing, img, original_img

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                img = original_img.copy()
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            img = original_img.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            coords = [ix, iy, x, y]
            if ix >= x and iy < y:
                coords = [x, iy, ix, y]
            elif ix >= x  and iy >= y:
                coords = [x, y, ix, iy]
            elif ix < x  and iy >= y:
                coords = [ix, y, x, iy]
            rectangle_coordinates.append(coords)
            original_img = img.copy()

    original_img = image.copy()
    img = original_img.copy()
    rectangle_coordinates = []

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_rectangle)

    while 1:
        cv2.imshow("image", img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cv2.destroyAllWindows()

    # returns a list of [x_start, y_start, x_end, y_end] rectangle coordinates
    return [np.where(x<0, 0, x) for x in np.array(rectangle_coordinates)]


def determine_threshold_colors_for_mask(image, rectangle_coordinates=[], save_mask=False, mask_output_path=None, invert=False):
    def doNothing(x):
        pass

    # creating a resizable window named Track Bars
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

    # creating track bars for gathering threshold values of red green and blue
    cv2.createTrackbar('min_blue', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('min_green', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('min_red', 'Track Bars', 0, 255, doNothing)

    cv2.createTrackbar('max_blue', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('max_green', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('max_red', 'Track Bars', 0, 255, doNothing)

    full_img = image.copy()

    # converting into HSV color model (first to gray to filter out weird blocky noise)
    hsv_image = cv2.cvtColor(cv2.cvtColor(cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

    # showing both resized and hsv image in named windows
    hsv_image_copy = hsv_image.copy()
    for coords in rectangle_coordinates:
        cv2.rectangle(full_img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
        cv2.rectangle(hsv_image_copy, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
    cv2.imshow('Base Image', full_img)
    cv2.imshow('HSV Image', hsv_image_copy)

    # creating a loop to get the feedback of the changes in trackbars
    while True:
        try:
            # reading the trackbar values for thresholds
            min_blue = cv2.getTrackbarPos('min_blue', 'Track Bars')
            min_green = cv2.getTrackbarPos('min_green', 'Track Bars')
            min_red = cv2.getTrackbarPos('min_red', 'Track Bars')

            max_blue = cv2.getTrackbarPos('max_blue', 'Track Bars')
            max_green = cv2.getTrackbarPos('max_green', 'Track Bars')
            max_red = cv2.getTrackbarPos('max_red', 'Track Bars')
        except:
            break

        # using inRange function to turn on the image pixels where object threshold is matched
        full_mask = cv2.inRange(hsv_image, (min_blue, min_green, min_red), (max_blue, max_green, max_red))

        if len(rectangle_coordinates) > 0:
            mask = np.zeros(full_mask.shape, np.uint8)
            for coords in rectangle_coordinates:
                mask[coords[1]:coords[3], coords[0]:coords[2]] = full_mask[coords[1]:coords[3], coords[0]:coords[2]]
        else:
            mask = full_mask

        if invert:
            mask = cv2.bitwise_not(mask)

        # showing the mask image
        cv2.imshow('Mask Image', mask)

        # checking if q key is pressed to break out of loop
        key = cv2.waitKey(25)
        if key == ord('q'):
            if save_mask:
                if mask_output_path is None:
                    print("could not save mask because mask_output_path was not given")
                else:
                    cv2.imwrite(mask_output_path, mask)
            break

    # printing the threshold values for usage in detection application
    print(f'min_blue {min_blue} min_green {min_green} min_red {min_red}')
    print(f'max_blue {max_blue} mmax_green {max_green} max_red {max_red}')

    # destroying all windows
    cv2.destroyAllWindows()

    return mask, (min_blue, min_green, min_red), (max_blue, max_green, max_red)
