import numpy as np
import cv2
import matplotlib.pyplot as plt

from main import image1_path, image2_path


def estimate_goodness_of_fit_laxi(matches, train_kp, query_kp):
    """
    Estimate the geometric transformation between matched keypoints using RANSAC.
    """
    train_pts = np.float32([train_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    query_pts = np.float32([query_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    transform, mask = cv2.estimateAffinePartial2D(train_pts, query_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    inliers = [m for m, msk in zip(matches, mask) if msk[0] == 1]
    inlier_frac = len(inliers) / len(matches) * 100 if len(matches) > 0 else 0

    return inlier_frac, inliers, mask


def calculate_match_properties(matches, train_kp, query_kp, image_width):
    """
    Calculate properties (length, angle) for each match.
    """
    properties = []

    for match in matches:
        query_pt = np.array(query_kp[match.queryIdx].pt)
        train_pt = np.array(train_kp[match.trainIdx].pt)
        train_pt_adjusted = np.array([train_pt[0] + image_width, train_pt[1]])

        length = np.sqrt(np.sum((query_pt - train_pt_adjusted) ** 2))
        dx = train_pt_adjusted[0] - query_pt[0]
        dy = train_pt_adjusted[1] - query_pt[1]
        angle = np.degrees(np.arctan2(-dy, dx))

        properties.append({
            'length': float(length),
            'angle': float(angle),
            'query_point': query_pt,
            'train_point': train_pt_adjusted
        })

    return properties


def calculate_spatial_distribution_score(match_properties, image_height, image_width):
    """
    Calculate how well distributed the matches are across the image.
    """
    if not match_properties:
        return 0.0

    query_points = np.array([prop['query_point'] for prop in match_properties])

    # Divide image into a 3x3 grid
    grid_h = image_height / 3
    grid_w = image_width / 3
    grid_counts = np.zeros((3, 3))

    # Count points in each grid cell
    for point in query_points:
        grid_x = min(2, int(point[0] / grid_w))
        grid_y = min(2, int(point[1] / grid_h))
        grid_counts[grid_y, grid_x] += 1

    total_points = len(query_points)
    expected_per_cell = total_points / 9

    # Calculate distribution metrics
    empty_cells = np.sum(grid_counts == 0)
    overloaded_cells = np.sum(grid_counts > 2 * expected_per_cell)

    # Calculate distribution score
    distribution_score = 1.0 - (empty_cells / 9) * 0.5 - (overloaded_cells / 9) * 0.5

    # Print distribution analysis
    print("\n=== Spatial Distribution Analysis ===")
    print(f"Empty grid cells: {empty_cells}/9")
    print(f"Overloaded cells: {overloaded_cells}/9")
    print(f"Distribution score: {distribution_score:.4f}")

    return max(0.0, distribution_score)


def calculate_statistics_with_tolerance(match_properties, max_image_dimension, tolerance=0.01):
    """
    Calculate statistical properties of matches.
    """
    if not match_properties:
        return {
            'length': {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0},
            'angle': {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        }

    lengths = [prop['length'] for prop in match_properties]
    angles = [prop['angle'] for prop in match_properties]

    length_stats = {
        'mean': float(np.mean(lengths)),
        'median': float(np.median(lengths)),
        'std': float(np.std(lengths)) if len(lengths) > 1 else 0,
        'min': float(np.min(lengths)),
        'max': float(np.max(lengths)),
        'count': len(lengths)
    }

    angle_stats = {
        'mean': float(np.mean(angles)),
        'median': float(np.median(angles)),
        'std': float(np.std(angles)) if len(angles) > 1 else 0,
        'min': float(np.min(angles)),
        'max': float(np.max(angles)),
        'count': len(angles)
    }

    return {'length': length_stats, 'angle': angle_stats}


def count_crossing_lines(matches, train_kp, query_kp, image_width, min_cross_angle=30):
    """
    Count and analyze crossing lines between matches.
    """

    def do_lines_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def calculate_line_angle(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.degrees(np.arctan2(dy, dx))

    def get_crossing_angle(line1_p1, line1_p2, line2_p1, line2_p2):
        angle1 = calculate_line_angle(line1_p1, line1_p2)
        angle2 = calculate_line_angle(line2_p1, line2_p2)
        diff = abs(angle1 - angle2) % 180
        return min(diff, 180 - diff)

    lines = []
    for m in matches:
        x1, y1 = query_kp[m.queryIdx].pt
        x2, y2 = train_kp[m.trainIdx].pt
        x2_shifted = x2 + image_width
        lines.append(((x1, y1), (x2_shifted, y2)))

    significant_crossings = 0
    total_angle_weighted_crossings = 0
    crossing_angles = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if do_lines_intersect(lines[i][0], lines[i][1], lines[j][0], lines[j][1]):
                crossing_angle = get_crossing_angle(lines[i][0], lines[i][1],
                                                    lines[j][0], lines[j][1])
                crossing_angles.append(crossing_angle)

                if crossing_angle > min_cross_angle:
                    significant_crossings += 1
                    angle_weight = (crossing_angle - min_cross_angle) / (180 - min_cross_angle)
                    total_angle_weighted_crossings += angle_weight

    crossing_fraction = significant_crossings / len(matches) if len(matches) > 0 else 0
    weighted_fraction = total_angle_weighted_crossings / len(matches) if len(matches) > 0 else 0

    print("\n=== Crossing Lines Analysis ===")
    print(f"Number of inliers: {len(matches)}")
    print(f"Total crossings: {len(crossing_angles)}")
    print(f"Significant crossings (>{min_cross_angle}°): {significant_crossings}")
    if crossing_angles:
        print(f"Crossing angle range: {min(crossing_angles):.1f}° - {max(crossing_angles):.1f}°")
        print(f"Mean crossing angle: {np.mean(crossing_angles):.1f}°")

    return weighted_fraction, significant_crossings


def calculate_LAX_FIT(length_std, length_mean, angle_std, crossing_fraction, num_inliers, total_matches,
                      min_inliers=10, image_properties=None):
    """
    Enhanced LAX-FIT calculation with spatial distribution and stronger penalties.
    """
    # Convert numpy values to scalars
    length_std = float(length_std) if hasattr(length_std, 'item') else length_std
    length_mean = float(length_mean) if hasattr(length_mean, 'item') else length_mean
    angle_std = float(angle_std) if hasattr(angle_std, 'item') else angle_std
    crossing_fraction = float(crossing_fraction) if hasattr(crossing_fraction, 'item') else crossing_fraction

    # Calculate basic metrics
    if length_mean != 0:
        length_ratio = length_std / length_mean
        F_length = max(0, 1 - (length_ratio / 0.2))
    else:
        length_ratio = 1.0
        F_length = 0.0

    F_angle = 1 if angle_std < 5 else max(0, 1 - (angle_std / 10))

    crossing_scale = max(1, 5 / num_inliers) if num_inliers > 0 else 1
    F_crossing = max(0, 1 - (crossing_fraction * 5 * crossing_scale))

    # Calculate inlier-based penalties
    inlier_ratio = num_inliers / total_matches if total_matches > 0 else 0

    # Stronger penalty for few inliers
    inlier_penalty = (num_inliers / min_inliers) ** 2 if num_inliers < min_inliers else 1.0

    # Stronger penalty for low inlier ratio
    # inlier_ratio_penalty = min(1.0, (inlier_ratio * 10) ** 0.5)
    inlier_ratio_penalty = 1

    # Calculate spatial distribution score
    if image_properties and 'match_properties' in image_properties:
        distribution_score = calculate_spatial_distribution_score(
            image_properties['match_properties'],
            image_properties['height'],
            image_properties['width']
        )
    else:
        distribution_score = 1.0

    # Updated weights
    w_length = 0.25
    w_angle = 0.25
    w_crossing = 0.2
    w_distribution = 0.3

    # Calculate base score
    base_score = (
            w_length * F_length +
            w_angle * F_angle +
            w_crossing * F_crossing +
            w_distribution * distribution_score
    )

    # Apply penalties
    final_score = base_score * inlier_penalty * inlier_ratio_penalty

    # Print detailed analysis
    print("\n=== LAX-FIT Analysis ===")
    print("Base Metrics:")
    print(f"  Length Score: {F_length:.4f}")
    print(f"  Angle Score: {F_angle:.4f}")
    print(f"  Crossing Score: {F_crossing:.4f}")
    print(f"  Distribution Score: {distribution_score:.4f}")

    print("\nPenalties:")
    print(f"  Inlier Count Penalty: {inlier_penalty:.4f}")
    print(f"  Inlier Ratio Penalty: {inlier_ratio_penalty:.4f}")
    print(f"  Inliers: {num_inliers}/{total_matches} ({inlier_ratio:.4f})")

    print("\nFinal Scores:")
    print(f"  Base Score: {base_score:.4f}")
    print(f"  Final Score: {final_score:.4f}")

    return final_score


def identify_with_LAX_FIT(query_image, train_image, visualize=False):
    """
    Main function to perform LAX-FIT analysis between two images.
    """
    # Convert images to grayscale if needed
    query_image_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY) if len(query_image.shape) == 3 else query_image
    train_image_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY) if len(train_image.shape) == 3 else train_image

    # Initialize SIFT detector and matcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Detect keypoints and compute descriptors
    query_keypoints, query_descriptors = sift.detectAndCompute(query_image_gray, None)
    train_keypoints, train_descriptors = sift.detectAndCompute(train_image_gray, None)

    print("\n=== Keypoint Detection ===")
    print(f"Query image keypoints: {len(query_keypoints)}")
    print(f"Train image keypoints: {len(train_keypoints)}")

    if query_descriptors is None or train_descriptors is None:
        print("No descriptors found in one or both images.")
        return 0.0

    # Match descriptors
    matches = bf.match(query_descriptors, train_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) == 0:
        print("No matches found between images.")
        return 0.0

    # Estimate geometric transformation
    score, inliers, mask = estimate_goodness_of_fit_laxi(matches, train_keypoints, query_keypoints)

    # Calculate match properties
    image_width = query_image.shape[1]
    inlier_properties = calculate_match_properties(inliers, train_keypoints, query_keypoints, image_width)

    # Calculate statistics
    max_image_dimension = max(query_image.shape[:2])
    stats = calculate_statistics_with_tolerance(inlier_properties, max_image_dimension)

    # Calculate crossing lines
    crossing_fraction, num_crossings = count_crossing_lines(inliers, train_keypoints, query_keypoints, image_width)

    # Prepare image properties for spatial distribution analysis
    image_properties = {
        'match_properties': inlier_properties,
        'height': query_image.shape[0],
        'width': query_image.shape[1]
    }

    # Calculate final LAX-FIT score
    LAX_FIT_score = calculate_LAX_FIT(
        stats['length']['std'],
        stats['length']['mean'],
        stats['angle']['std'],
        crossing_fraction,
        len(inliers),
        len(matches),
        image_properties=image_properties
    )

    # Visualize matches if requested
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
    print("\n=== Matches Length Statistics ===")
    print(f"Mean length: {stats['length']['mean']:.2f}")
    print(f"Median length: {stats['length']['median']:.2f}")
    print(f"Length std dev: {stats['length']['std']:.2f}")
    print(f"Length range: {stats['length']['min']:.2f} - {stats['length']['max']:.2f}")
    print(f"Number of length measurements: {stats['length']['count']}")

    print("\n=== Matches Angle Statistics ===")
    print(f"Mean angle: {stats['angle']['mean']:.2f}°")
    print(f"Median angle: {stats['angle']['median']:.2f}°")
    print(f"Angle std dev: {stats['angle']['std']:.2f}°")
    print(f"Angle range: {stats['angle']['min']:.2f}° - {stats['angle']['max']:.2f}°")
    print(f"Number of angle measurements: {stats['angle']['count']}")

    print("\n=== Matches Statistics ===")
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
image1_path = r'path_image_1.jpg'
image2_path = r'path_image-2.jpg'

# Load images
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# Run the LAX-FIT identification with visualization
identify_with_LAX_FIT(img1, img2, visualize=True)
