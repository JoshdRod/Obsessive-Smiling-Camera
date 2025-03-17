import cv2
import numpy as np
import math
import os
import glob
from threading import Timer
import time

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
smileBuffer = [] # Stores all the smiles found in a 5 second window

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

save_path = r"C:\Users\joshr\OneDrive\Documents\Projects\Python Projects\Obsessive-Smiling-Camera\Ranked Smiles"
if not os.path.exists(save_path):
    os.makedirs(save_path)

class SmileSaveTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            saveTopFiveSmiles(smileBuffer)
            print("Saved smiles! Now waiting 5 secs")


"""
Add smile to the smile buffer (stores smiles from past 5 seconds, sorted by intensity) (used for saving top smiles every 5 secs)
INPUT: image, float rating
OUTPUT: dict {image, rating} to smile buffer, sorted
"""
def addToSmileBuffer(smiling_image: list, smile_rating: float):
    smileEntry = {"image": smiling_image,
                  "rating": smile_rating}
    smileBuffer.append(smileEntry)
    smileBuffer.sort(reverse=True, key=lambda x: x["rating"]) # Sort from best to worst smile rating (inefficient, but shouldn't be an issue for small arr sizes)
    print("Added smile to buffer!")

"""
Save Top 5 smiles in smile buffer to folder
INPUT: global smile buffer
OUTPUT: writes image to folder
"""
def saveTopFiveSmiles(smileBuffer):
    for i in range(5):
        if len(smileBuffer) <= i:
            print(f"Saved top {i} images in last 5 seconds to {save_path}")
            break
        file_path = os.path.join(save_path, f"Smile{i}.jpg")
        cv2.imwrite(file_path, smileBuffer[i]["image"])
    smileBuffer = []


"""
Takes in frame, finds smiles, adds them to 5 second buffer in order of smile intensity
INPUT: image frame
OUTPUTS: image frame and int intensity into smile buffer
"""
def process_frame(frame):
    greyscaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(greyscaleImage, scaleFactor=1.1, minNeighbors=12, minSize=(30, 30))

    for person_id, (x, y, w, h) in enumerate(faces):
        faceImage = greyscaleImage[y:y + h, x:x + w]
        smilesInImage = smileCascade.detectMultiScale(faceImage, scaleFactor=1.8, minNeighbors=35, minSize=(25, 25))

        if len(smilesInImage) == 0:
            break

        for (sx, sy, sw, sh) in smilesInImage:
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)
            widthRatio = sx / x
            smile_intensity = 1 - math.exp(-(10 * widthRatio))
            smile_rating = round(smile_intensity * 10, 2)
            smiling_image = frame[y:y + h, x:x + w]
            addToSmileBuffer(smiling_image, smile_rating)
            break

def show_top_5_images():
    images = glob.glob(os.path.join(save_path, "smiling_face_*.jpg"))
    image_ratings = []

    for img_path in images:
        file_name = os.path.basename(img_path)
        rating = float(file_name.split("_")[-1].split(".")[0]) 
        image_ratings.append((img_path, rating))

    image_ratings.sort(key=lambda x: x[1], reverse=True)
    top_5 = image_ratings[:5]

    for i, (img_path, rating) in enumerate(top_5):
        image = cv2.imread(img_path)
        cv2.putText(image, f"Smile Intensity: {rating}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Top {i+1} - Rating: {rating}/10", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

def show_worst_smiler():
    images = glob.glob(os.path.join(save_path, "smiling_face_*.jpg"))
    image_ratings = []

    for img_path in images:
        file_name = os.path.basename(img_path)
        rating = float(file_name.split("_")[-1].split(".")[0]) 
        image_ratings.append((img_path, rating))

    image_ratings.sort(key=lambda x: x[1])
    worst_smiler = image_ratings[0]  # The one with the lowest rating

    img_path, rating = worst_smiler
    image = cv2.imread(img_path)
    cv2.putText(image, f"Smile Intensity: {rating}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(f"Worst Smiler - Rating: {rating}/10", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_button(window, text, position, callback):
    font = cv2.FONT_HERSHEY_SIMPLEX
    button_width = 200
    button_height = 50
    button_color = (200, 200, 255)
    text_color = (0, 0, 0)

    button_img = np.ones((button_height, button_width, 3), dtype=np.uint8) * 255
    cv2.putText(button_img, text, (10, button_height // 2 + 10), font, 1, text_color, 2)

    x, y = position
    window[y:y+button_height, x:x+button_width] = button_img

    return (x, y, button_width, button_height, callback)

def mouse_callback(event, x, y, flags, param):
    window, buttons = param
    if event == cv2.EVENT_LBUTTONDOWN:
        for (bx, by, bw, bh, callback) in buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                callback()

def main_menu():
    window_name = "Main Menu"
    window = np.ones((600, 600, 3), dtype=np.uint8) * 255
    buttons = [
        draw_button(window, "Start Smile Detection", (200, 100), main_loop),
        draw_button(window, "View Top 5 Highest-Rated Smiles", (200, 200), show_top_5_images),
        draw_button(window, "View Worst Smiler", (200, 300), show_worst_smiler),
        draw_button(window, "Exit", (200, 400), exit_program)
    ]

    cv2.imshow(window_name, window)
    cv2.setMouseCallback(window_name, mouse_callback, param=(window, buttons))

    while True:
        cv2.waitKey(1)
    cv2.destroyAllWindows()

def main_loop():
    smileSaveTimer = SmileSaveTimer(5, None)
    smileSaveTimer.start()
    while True:
        ret, frame = cam.read()
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)

        process_frame(frame)
        cv2.imshow("Image", frame)

        key = cv2.waitKey(1)
        if key == 27:
            exit_program()

def exit_program():
    print("Exiting program.")
    cam.release()
    cv2.destroyAllWindows()
    exit()

main_menu()
