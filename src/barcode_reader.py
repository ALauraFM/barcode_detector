import cv2
import math
from pyzbar import pyzbar
import numpy as np
import pyrealsense2 as rs

# Count how many paralel lines are in the image
def findparallel(lines):
    lines1 = []
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue
            if abs(lines[i][0][1] - lines[j][0][1]) == 0:
                lines1.append((i, j))
    return len(lines1)

# Counts how many lines there are in the image
def detect_lines(img):
    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blurs the image
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # Detect edges
    low_threshold = 30
    high_threshold = 100
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Gets the smallest side of the image. This is used to find lines proportional to the image size
    size_img = img.shape
    min_size = min(img.shape[0], img.shape[1])

    # Uses the Hough Transform to find lines in the edge-detected image
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = math.ceil(min_size * 0.1)  # minimum number of pixels making up a line
    max_line_gap = 1  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    parallel = 0

    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    if lines is not None:
        parallel = findparallel(lines)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    if lines is not None:
        return parallel, lines_edges
    else:
        return 0, img

# Make one method to decode the barcode
def barcodeReader(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret,th3 = cv2.threshold(gray,110,255,cv2.THRESH_BINARY)
    #uses pyzbar to decode barcode
    detectedBarcodes = pyzbar.decode(th3)

    barcodeData = None
    # If detected then print the message
    if detectedBarcodes:
        # Traverse through all the detected barcodes in image
        for barcode in detectedBarcodes:

            # Locate the barcode position in image
            (x, y, w, h) = barcode.rect

            # Put the rectangle in image using
            # cv2 to highlight the barcode
            cv2.rectangle(img, (x-10, y-10),
                          (x + w+10, y + h+10),
                          (255, 0, 0), 2)

            if barcode.data != "":
                # barcode.data É BINÁRIO
                barcodeData = barcode.data
       
        print(barcodeData)
        return barcodeData

def displayCode(code, image, x, y):
    text = code
    # Barcode coordinates referred to the image
    coordinates = (x, y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 4
    color = (0, 0, 255)
    thickness = 5
    image = cv2.putText(image, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    return image

def BarcodeFinder():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    pipeline.start(config)
    
    try:
        while True:
            frame = pipeline.wait_for_frames()
            color_frame = frame.get_color_frame()
            
            if not color_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            #In case the camera is positioned sideways, uncomment the next line
            #color_image = cv2.rotate(color_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # calculate x & y gradient
            gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

            # subtract the y-gradient from the x-gradient
            gradient = cv2.subtract(gradX, gradY)
            gradient = cv2.convertScaleAbs(gradient)

            # blur the image
            blurred = cv2.blur(gradient, (2, 2))

            # threshold the image
            (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

            # construct a closing kernel and apply it to the thresholded image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            # perform a series of erosions and dilations
            closed = cv2.erode(closed, None, iterations=5)
            closed = cv2.dilate(closed, None, iterations=5)


            # find the contours in the thresholded image, then sort the contours
            # by their area, keeping only the largest one
            cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            # Inicializa a variável dentro do loop para garantir que ela seja definida mesmo que nenhum contorno seja encontrado
            image_with_contours = color_image.copy()

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                ratio = float(w) / h
                
                # BARCODE IS 10 x 6 (w x h). ASSUMING THE BARCODE IS HORIZONTAL
                
                if ratio > 0.3 and ratio < 5:
                    rect = cv2.minAreaRect(c)
                    box = np.intp(cv2.boxPoints(rect))
                    
                    max_point = box.argmax(axis=0)
                    min_point = box.argmin(axis=0)

                    # PADDING TO THE BARCODE (BARCODE AND SURROUNDING)
                    offset_x = 15
                    offset_y = 8

                    x1 = box[min_point[0]][0] - offset_x
                    y1 = box[min_point[1]][1] - offset_y

                    x2 = box[max_point[0]][0] + offset_x
                    y2 = box[max_point[1]][1] + offset_y

                    im_box = color_image[y1:y2, x1:x2]
                    im_box_shape = im_box.shape
                    
                    
                    # IGNORE SMALL RECTANGLES
                    if (im_box_shape[0] > 50 or im_box_shape[1] > 50) and im_box_shape[0] > 0 and im_box_shape[1] > 0:
                    
                        # COUNT # OF PARALLEL LINES IN IMAGE. IF IT'S HIGH, IT'S PROBABLY A BARCODE
                        
                        number_of_parallel_lines, img_with_lines = detect_lines(im_box)
                        
                        if number_of_parallel_lines > 50:
                            # DECODE THE BARCODE
                            code = barcodeReader(im_box)
                            if code is not None:
                                # Draw the contour on the copy of the image
                                cv2.drawContours(img_with_lines, [box], -1, (0, 255, 0), 3)
                                image_with_contours = displayCode(code.decode(), img_with_lines, x, y)

            # Shows the image with it's contours
            cv2.imshow("Image with Contours", image_with_contours)
            #The loop exits when the 'q' key is pressed. 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    BarcodeFinder()
