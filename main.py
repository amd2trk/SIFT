import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from google.colab.patches import cv2_imshow
import os

def process_image(query_img_path, target_img_path):
    # Load images
    query_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    target_img_gray = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    target_img_color = cv2.imread(target_img_path)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp_query, des_query = sift.detectAndCompute(query_img, None)
    kp_target, des_target = sift.detectAndCompute(target_img_gray, None)

    # Create BFMatcher and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des_query, des_target, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    # Get matched keypoints in the target image
    matched_kp = [kp_target[m.trainIdx] for m in good_matches]

    # Draw keypoints on the target image
    result_img = cv2.drawKeypoints(target_img_color, matched_kp, None, 
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display result
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('Object Location in Target Image')
    plt.axis('off')
    plt.show()

    # Save result
    cv2.imwrite('result.png', result_img)
    print("Result saved as 'result.png'")
    files.download('result.png')

def process_video(query_img_path, video_path):
    # Load query image and initialize SIFT
    query_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp_query, des_query = sift.detectAndCompute(query_img, None)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video writer
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors in frame
        kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)

        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des_query, des_frame, k=2)

        # Apply ratio test
        good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

        if len(good_matches) > 4:  # Minimum matches for a bounding box
            # Get matched keypoints' coordinates
            pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches])
            
            # Compute bounding box
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            
            # Draw rectangle
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

    # Clean up
    cap.release()
    out.release()
    print("Processed video saved as 'output_video.mp4'")
    files.download('output_video.mp4')

def main():
    # Upload files
    print("Upload query image (query.jpg):")
    query_upload = files.upload()
    query_img_path = list(query_upload.keys())[0]

    mode = input("Enter mode ('image' or 'video'): ").strip().lower()

    if mode == 'image':
        print("Upload target image (target.jpg):")
        target_upload = files.upload()
        target_img_path = list(target_upload.keys())[0]
        process_image(query_img_path, target_img_path)
    elif mode == 'video':
        print("Upload test video (test_video.mp4):")
        video_upload = files.upload()
        video_path = list(video_upload.keys())[0]
        process_video(query_img_path, video_path)
    else:
        print("Invalid mode. Choose 'image' or 'video'.")

if __name__ == '__main__':
    main()
