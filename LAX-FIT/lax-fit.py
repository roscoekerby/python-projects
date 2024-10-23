import numpy as np
import cv2
import matplotlib.pyplot as plt

def estimate_goodness_of_fit(matches, train_kp, query_kp):
    # Convert keypoints to numpy arrays
    train_pts = np.float32([train_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    query_pts = np.float32([query_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate geometric transformation using RANSAC
    transform, mask = cv2.estimateAffinePartial2D(train_pts, query_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    # Filter matches based on the RANSAC mask
    inliers = [m for m, msk in zip(matches, mask) if msk[0] == 1]

    # Calculate the percentage of inliers
    inlier_frac = len(inliers) / len(matches) * 100

    return inlier_frac, inliers, mask

def calculate_match_properties(matches, train_kp, query_kp):
    properties = []
    for match in matches:
        # Get the keypoints
        query_pt = np.array(query_kp[match.queryIdx].pt)
        train_pt = np.array(train_kp[match.trainIdx].pt)

        # Calculate length (distance between points)
        length = np.linalg.norm(query_pt - train_pt)

        # Calculate angle between points (in degrees)
        angle = np.arctan2(train_pt[1] - query_pt[1], train_pt[0] - query_pt[0]) * 180.0 / np.pi

        # Append properties (length, angle)
        properties.append({
            'length': length,
            'angle': angle,
            'query_point': query_pt,
            'train_point': train_pt
        })

    return properties

def calculate_statistics_with_tolerance(match_properties, max_image_dimension, tolerance=0.01):
    # Extract lengths and angles
    lengths = [prop['length'] for prop in match_properties]
    angles = [prop['angle'] for prop in match_properties]

    # Filter out values close to the mean
    lengths = [l for l in lengths if abs(l - np.mean(lengths)) > tolerance]
    angles = [a for a in angles if abs(a - np.mean(angles)) > tolerance]

    # Calculate length statistics
    length_stats = {
        'mean': np.mean(lengths) if lengths else 0,
        'median': np.median(lengths) if lengths else 0,
        'std': np.std(lengths) if lengths else 0,
        'min': np.min(lengths) if lengths else 0,
        'max': np.max(lengths) if lengths else 0,
        'count': len(lengths)
    }

    # Calculate angle statistics
    angle_stats = {
        'mean': np.mean(angles) if angles else 0,
        'median': np.median(angles) if angles else 0,
        'std': np.std(angles) if angles else 0,
        'min': np.min(angles) if angles else 0,
        'max': np.max(angles) if angles else 0,
        'count': len(angles)
    }

    return {
        'length': length_stats,
        'angle': angle_stats
    }

def do_lines_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    return (ccw(p1,p3,p4) != ccw(p2,p3,p4)) and (ccw(p1,p2,p3) != ccw(p1,p2,p4))

def count_crossing_lines(matches, train_kp, query_kp, image_width):
    lines = []
    for m in matches:
        x1, y1 = query_kp[m.queryIdx].pt
        x2, y2 = train_kp[m.trainIdx].pt
        x2_shifted = x2 + image_width
        lines.append(((x1, y1), (x2_shifted, y2)))

    num_crossings = 0
    total_pairs = len(lines) * (len(lines) - 1) / 2

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            if do_lines_intersect(line1[0], line1[1], line2[0], line2[1]):
                num_crossings += 1

    crossing_fraction = num_crossings / total_pairs if total_pairs > 0 else 0

    return crossing_fraction, num_crossings

def calculate_LAX_FIT(length_std, length_mean, angle_std, crossing_fraction):
    # Convert numpy values to scalars if they're numpy arrays
    length_std = float(length_std) if hasattr(length_std, 'item') else length_std
    length_mean = float(length_mean) if hasattr(length_mean, 'item') else length_mean
    angle_std = float(angle_std) if hasattr(angle_std, 'item') else angle_std
    crossing_fraction = float(crossing_fraction) if hasattr(crossing_fraction, 'item') else crossing_fraction

    # Calculate normalized metrics
    F_length = 1 - (length_std / length_mean if length_mean != 0 else 0)
    F_angle = 1 - (angle_std / 180)  # Normalized over 180 degrees
    F_crossing = 1 - crossing_fraction

    # Define weights
    w_length = 0.3
    w_angle = 0.3
    w_crossing = 0.4

    # Verify weights sum to 1
    total_weight = w_length + w_angle + w_crossing
    if not np.isclose(total_weight, 1.0):
        print(f"Warning: Weights sum to {total_weight}, but should sum to 1. Adjusting weights.")
        w_length /= total_weight
        w_angle /= total_weight
        w_crossing /= total_weight

    # Calculate contributions
    contribution_length = w_length * F_length
    contribution_angle = w_angle * F_angle
    contribution_crossing = w_crossing * F_crossing

    # Calculate final score
    LAX_FIT_score = contribution_length + contribution_angle + contribution_crossing

    # Print calculation details
    print("\n--- LAX-FIT Calculation Details ---")
    print(f"Normalized Length Consistency (F_length): {F_length:.4f} (Weight: {w_length:.4f}) => Contribution: {contribution_length:.4f}")
    print(f"Normalized Angle Consistency (F_angle): {F_angle:.4f} (Weight: {w_angle:.4f}) => Contribution: {contribution_angle:.4f}")
    print(f"Normalized Crossing Fraction (F_crossing): {F_crossing:.4f} (Weight: {w_crossing:.4f}) => Contribution: {contribution_crossing:.4f}")
    print(f"Total LAX-FIT Score: {LAX_FIT_score:.4f}")
    print("--------------------------------\n")

    return LAX_FIT_score

def identify_with_LAX_FIT(query_image, train_image, visualize=False):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    query_image_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY) if len(query_image.shape) == 3 else query_image
    train_image_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY) if len(train_image.shape) == 3 else train_image

    query_keypoints, query_descriptors = sift.detectAndCompute(query_image_gray, None)
    train_keypoints, train_descriptors = sift.detectAndCompute(train_image_gray, None)

    print("\n=== Keypoint Detection ===")
    print(f"Query image keypoints: {len(query_keypoints)}")
    print(f"Train image keypoints: {len(train_keypoints)}")

    matches = bf.match(query_descriptors, train_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    score, inliers, mask = estimate_goodness_of_fit(matches, train_keypoints, query_keypoints)
    inlier_properties = calculate_match_properties(inliers, train_keypoints, query_keypoints)

    max_image_dimension = max(query_image.shape[:2])
    stats = calculate_statistics_with_tolerance(inlier_properties, max_image_dimension)

    image_width = query_image.shape[1]
    crossing_fraction, num_crossings = count_crossing_lines(inliers, train_keypoints, query_keypoints, image_width)

    LAX_FIT_score = calculate_LAX_FIT(stats['length']['std'], stats['length']['mean'],
                                     stats['angle']['std'], crossing_fraction)

    if visualize:
        img_matches = cv2.drawMatches(query_image, query_keypoints, train_image, train_keypoints,
                                    matches[:50], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img_inliers = cv2.drawMatches(query_image, query_keypoints, train_image, train_keypoints,
                                    inliers[:50], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.title('Matches (Before RANSAC)')
        plt.imshow(img_matches)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Inliers (After RANSAC)')
        plt.imshow(img_inliers)
        plt.axis('off')

        plt.show()

    # Print results in organized groups
    print("\n=== Length Statistics ===")
    print(f"Mean length: {stats['length']['mean']:.2f}")
    print(f"Median length: {stats['length']['median']:.2f}")
    print(f"Length std dev: {stats['length']['std']:.2f}")
    print(f"Length range: {stats['length']['min']:.2f} - {stats['length']['max']:.2f}")
    print(f"Number of length measurements: {stats['length']['count']}")

    print("\n=== Angle Statistics ===")
    print(f"Mean angle: {stats['angle']['mean']:.2f}°")
    print(f"Median angle: {stats['angle']['median']:.2f}°")
    print(f"Angle std dev: {stats['angle']['std']:.2f}°")
    print(f"Angle range: {stats['angle']['min']:.2f}° - {stats['angle']['max']:.2f}°")
    print(f"Number of angle measurements: {stats['angle']['count']}")

    print("\n=== Match Statistics ===")
    print(f"Total matches: {len(matches)}")
    print(f"Inliers: {len(inliers)}")
    print(f"Outliers: {len(matches) - len(inliers)}")
    print(f"Crossing lines: {num_crossings}")
    print(f"Crossing fraction: {crossing_fraction:.4f}")

    print("\n=== Final Scores ===")
    print(f"Goodness of Fit Score: {score:.2f}%")
    print(f"LAX-FIT Score: {LAX_FIT_score:.2f}")

    return LAX_FIT_score
# Example usage:
image1_path = 'image1.jpg'
image2_path = 'image2.jpg'

# Load images
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# Run the LAX-FIT identification with visualization
identify_with_LAX_FIT(img1, img2, visualize=True)
