import pymysql
import os
import cv2

target_directory = "C:/Users/abdel/Downloads/SOCOFing/Real"

def find_fingerprint_match(source_image_path, target_filenames):
    source_image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)

    if source_image is None:
        raise ValueError(f"Cannot read image from path: {source_image_path}")

    best_score = 0
    best_match_filename = None

    for file in target_filenames:
        if file is not None:
            file_with_extension = file + ".bmp"
            target_image = cv2.imread(os.path.join(target_directory, file_with_extension), cv2.IMREAD_GRAYSCALE)

            if target_image is None:
                print(f"Cannot read image from path: {os.path.join(target_directory, file_with_extension)}")
                continue

            sift = cv2.SIFT.create()
            kp1, des1 = sift.detectAndCompute(source_image, None)
            kp2, des2 = sift.detectAndCompute(target_image, None)
            matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict()).knnMatch(des1, des2, k=2)

            match_points = []
            for p, q in matches:
                if p.distance < 0.1 * q.distance:
                    match_points.append(p)

            keypoints = min(len(kp1), len(kp2))
            score = len(match_points) / keypoints * 100

            if score > best_score:
                best_score = score
                best_match_filename = file
                
           
                result = cv2.drawMatches(source_image, kp1, target_image, kp2, match_points, None)
                result = cv2.resize(result, None, fx=2.5, fy=2.5)
                cv2.imshow("result", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
       
    is_match_found = best_score > 0
    return is_match_found, best_score, best_match_filename
