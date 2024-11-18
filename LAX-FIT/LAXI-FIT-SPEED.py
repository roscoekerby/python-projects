import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy.typing as npt

@dataclass
class MatchProperties:
    length: float
    angle: float
    query_point: npt.NDArray
    train_point: npt.NDArray

@dataclass
class ImageProperties:
    match_properties: List[MatchProperties]
    height: int
    width: int

@dataclass
class Statistics:
    mean: float
    median: float
    std: float
    min: float
    max: float
    count: int

def estimate_goodness_of_fit_laxi(matches: List, train_kp: List, query_kp: List) -> Tuple[float, List, npt.NDArray]:
    if not matches:
        return 0.0, [], np.array([])
    
    train_pts = np.float32([train_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    query_pts = np.float32([query_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    transform, mask = cv2.estimateAffinePartial2D(train_pts, query_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    mask = mask.ravel()
    inliers = [m for m, msk in zip(matches, mask) if msk]
    inlier_frac = len(inliers) / len(matches) * 100 if matches else 0

    return inlier_frac, inliers, mask

def calculate_match_properties(matches: List, train_kp: List, query_kp: List, image_width: int) -> List[MatchProperties]:
    properties = []
    
    query_pts = np.array([query_kp[m.queryIdx].pt for m in matches])
    train_pts = np.array([train_kp[m.trainIdx].pt for m in matches])
    train_pts_adjusted = train_pts + np.array([image_width, 0])
    
    lengths = np.sqrt(np.sum((query_pts - train_pts_adjusted) ** 2, axis=1))
    dx = train_pts_adjusted[:, 0] - query_pts[:, 0]
    dy = train_pts_adjusted[:, 1] - query_pts[:, 1]
    angles = np.degrees(np.arctan2(-dy, dx))
    
    for i in range(len(matches)):
        properties.append(MatchProperties(
            float(lengths[i]),
            float(angles[i]),
            query_pts[i],
            train_pts_adjusted[i]
        ))
    
    return properties

def calculate_spatial_distribution_score(match_properties: List[MatchProperties], 
                                      image_height: int, 
                                      image_width: int) -> float:
    if not match_properties:
        return 0.0

    query_points = np.array([prop.query_point for prop in match_properties])
    grid_x = np.minimum(2, (query_points[:, 0] / (image_width / 3)).astype(int))
    grid_y = np.minimum(2, (query_points[:, 1] / (image_height / 3)).astype(int))
    
    grid_counts = np.zeros((3, 3))
    np.add.at(grid_counts, (grid_y, grid_x), 1)
    
    expected_per_cell = len(query_points) / 9
    empty_cells = np.sum(grid_counts == 0)
    overloaded_cells = np.sum(grid_counts > 2 * expected_per_cell)
    
    distribution_score = 1.0 - (empty_cells / 9) * 0.5 - (overloaded_cells / 9) * 0.5
    return max(0.0, distribution_score)

def calculate_statistics(values: npt.NDArray) -> Statistics:
    if not len(values):
        return Statistics(0, 0, 0, 0, 0, 0)
    
    return Statistics(
        float(np.mean(values)),
        float(np.median(values)),
        float(np.std(values)) if len(values) > 1 else 0,
        float(np.min(values)),
        float(np.max(values)),
        len(values)
    )

def count_crossing_lines(matches: List, train_kp: List, query_kp: List, 
                        image_width: int, min_cross_angle: float = 30) -> Tuple[float, int]:
    if len(matches) < 2:
        return 0.0, 0

    p1 = np.array([query_kp[m.queryIdx].pt for m in matches])
    p2 = np.array([train_kp[m.trainIdx].pt for m in matches]) + np.array([image_width, 0])
    
    directions = p2 - p1
    angles = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
    
    significant_crossings = 0
    total_angle_weighted = 0.0

    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            v1 = np.hstack([p2[i] - p1[i], [0]])
            v2 = np.hstack([p2[j] - p1[j], [0]])
            cross_prod = np.cross(v1, v2)
            
            if abs(cross_prod[-1]) > 1e-10:
                angle_diff = abs(angles[i] - angles[j]) % 180
                crossing_angle = min(angle_diff, 180 - angle_diff)
                
                if crossing_angle > min_cross_angle:
                    significant_crossings += 1
                    angle_weight = (crossing_angle - min_cross_angle) / (180 - min_cross_angle)
                    total_angle_weighted += angle_weight
    
    return (total_angle_weighted / len(matches) if matches else 0.0), significant_crossings

def calculate_LAXI_FIT(length_std: float, length_mean: float, angle_std: float, 
                      crossing_fraction: float, num_inliers: int, total_matches: int,
                      min_inliers: int = 10, image_properties: Optional[Dict] = None) -> float:
    
    length_ratio = length_std / length_mean if length_mean != 0 else 1.0
    F_length = max(0, 1 - (length_ratio / 0.2))
    F_angle = 1 if angle_std < 5 else max(0, 1 - (angle_std / 10))
    
    crossing_scale = max(1, 5 / num_inliers) if num_inliers > 0 else 1
    F_crossing = max(0, 1 - (crossing_fraction * 5 * crossing_scale))
    
    distribution_score = (
        calculate_spatial_distribution_score(
            image_properties['match_properties'],
            image_properties['height'],
            image_properties['width']
        )
        if image_properties and 'match_properties' in image_properties
        else 1.0
    )
    
    weights = np.array([0.25, 0.25, 0.2, 0.3])
    scores = np.array([F_length, F_angle, F_crossing, distribution_score])
    base_score = np.dot(weights, scores)
    
    final_score = min(base_score, 0.5 * (num_inliers / min_inliers) * base_score) if num_inliers < min_inliers else base_score
    
    return final_score

def identify_with_LAXI_FIT(query_image: npt.NDArray, train_image: npt.NDArray, 
                          visualize: bool = False) -> float:
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY) if len(query_image.shape) == 3 else query_image
    train_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY) if len(train_image.shape) == 3 else train_image
    
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    query_kp, query_desc = sift.detectAndCompute(query_gray, None)
    train_kp, train_desc = sift.detectAndCompute(train_gray, None)
    
    if query_desc is None or train_desc is None:
        return 0.0
    
    matches = sorted(bf.match(query_desc, train_desc), key=lambda x: x.distance)
    if not matches:
        return 0.0
    
    score, inliers, mask = estimate_goodness_of_fit_laxi(matches, train_kp, query_kp)
    inlier_properties = calculate_match_properties(inliers, train_kp, query_kp, query_image.shape[1])
    
    lengths = np.array([prop.length for prop in inlier_properties])
    angles = np.array([prop.angle for prop in inlier_properties])
    length_stats = calculate_statistics(lengths)
    angle_stats = calculate_statistics(angles)
    
    crossing_fraction, num_crossings = count_crossing_lines(inliers, train_kp, query_kp, query_image.shape[1])
    
    image_props = {
        'match_properties': inlier_properties,
        'height': query_image.shape[0],
        'width': query_image.shape[1]
    }
    
    LAXI_FIT_score = calculate_LAXI_FIT(
        length_stats.std,
        length_stats.mean,
        angle_stats.std,
        crossing_fraction,
        len(inliers),
        len(matches),
        image_properties=image_props
    )
    
    if visualize:
        img_matches = cv2.drawMatches(
            query_image, query_kp, train_image, train_kp,
            matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow('Matches', img_matches)
        cv2.waitKey(1)
    
    return LAXI_FIT_score
