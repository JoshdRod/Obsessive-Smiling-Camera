import cv2
import numpy as np
import math
import os
import glob

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

save_path = r"C:\Users\simmo\OneDrive\Smile Pictures"
if not os.path.exists(save_path):
    os.makedirs(save_path)

def save_and_show_smile_image(smiling_image, smile_rating, person_id):
    file_path = os.path.join(save_path, f"smiling_face_{person_id}_{smile_rating}.jpg")
    cv2.imwrite(file_path, smiling_image)
    print(f"Smiling face saved to: {file_path}")
    cv2.putText(smiling_image, f"Smile Intensity: {smile_rating}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return smiling_image

def process_frame(frame):
    greyscaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(greyscaleImage, scaleFactor=1.1, minNeighbors=12, minSize=(30, 30))

    smile_detected = True
    smiling_people = []
    ratings = []

    for person_id, (x, y, w, h) in enumerate(faces):
        faceImage = greyscaleImage[y:y + h, x:x + w]
        smilesInImage = smileCascade.detectMultiScale(faceImage, scaleFactor=1.8, minNeighbors=35, minSize=(25, 25))

        if len(smilesInImage) == 0:
            smile_detected = False
            break

        for (sx, sy, sw, sh) in smilesInImage:
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)
            widthRatio = sx / x
            smile_intensity = 1 - math.exp(-(10 * widthRatio))
            smile_rating = round(smile_intensity * 10, 2)
            smiling_image = frame[y:y + h, x:x + w]
            saved_image = save_and_show_smile_image(smiling_image, smile_rating, person_id)
            smiling_people.append(saved_image)
            ratings.append(smile_rating)
            break

    return smile_detected, smiling_people, ratings

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
    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        smile_detected, smiling_people, ratings = process_frame(frame)

        if smile_detected:
            for i, smiling_image in enumerate(smiling_people):
                cv2.imshow(f"Person {i+1} - Smile Rating: {ratings[i]}/10", smiling_image)

            print("All faces smiled! Press 'Esc' to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Not all faces are smiling. Please try again.")

        cv2.imshow("Detected Faces", frame)

        key = cv2.waitKey(1)
        if key == 27:
            print("Exiting program.")
            break
        elif key == ord('q'):
            print("Exiting program.")
            break

def exit_program():
    print("Exiting program.")
    cam.release()
    cv2.destroyAllWindows()
    exit()

main_menu()
