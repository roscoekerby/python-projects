# LAX-FIT Algorithm

LAX-FIT (Length-Angle-Crossing Feature-based Image Testing) is a sophisticated image matching algorithm that evaluates the quality of feature matches between two images based on geometric properties of the matching keypoints. It combines multiple metrics including match lengths, angles, crossing patterns, and RANSAC inlier analysis to produce a comprehensive similarity score.

## Features

- **Multi-metric Analysis**: Evaluates matches based on length consistency, angle patterns, and line crossings
- **RANSAC-based Filtering**: Uses RANSAC to filter out outlier matches
- **Comprehensive Statistics**: Calculates detailed statistics for match properties
- **Visualization Support**: Optional visualization of matches before and after RANSAC filtering
- **Robust Scoring**: Adaptive scoring system considering multiple geometric properties
- **Debug Information**: Detailed debugging output for match analysis

## Requirements

```bash
pip install numpy
pip install opencv-python
pip install matplotlib
```

## Core Components

### 1. Geometric Analysis Functions

- `estimate_goodness_of_fit()`: RANSAC-based match filtering
- `calculate_match_properties()`: Computes geometric properties of matches
- `calculate_statistics_with_tolerance()`: Statistical analysis of match properties
- `count_crossing_lines()`: Analyzes line intersection patterns

### 2. LAX-FIT Score Calculation

```python
def calculate_LAX_FIT(length_std, length_mean, angle_std, crossing_fraction, num_inliers, total_matches, min_inliers=10):
    # Early exit for insufficient matches
    if total_matches == 0 or num_inliers < min_inliers:
        return 0.0

    # Calculate normalized metrics
    F_length = 1 - min((length_std / length_mean if length_mean != 0 else 1), 1)
    F_angle = 1 - min((angle_std / 45), 1)
    F_crossing = 1 - min(crossing_fraction * 2, 1)
    F_inlier = min((num_inliers / total_matches) * 3, 1)

    # Weighted combination
    score = (0.33 * F_length + 
            0.33 * F_angle + 
            0.33 * F_crossing + 
            0.01 * F_inlier)

    return score
```

### 3. Main Identification Function

```python
def identify_with_LAX_FIT(query_image, train_image, visualize=False):
    # Initialize SIFT detector and matcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Detect keypoints and compute descriptors
    query_keypoints, query_descriptors = sift.detectAndCompute(query_image_gray, None)
    train_keypoints, train_descriptors = sift.detectAndCompute(train_image_gray, None)

    # Match features and calculate LAX-FIT score
    matches = bf.match(query_descriptors, train_descriptors)
    # ... score calculation ...
    return LAX_FIT_score
```

## Usage

### Basic Usage

```python
import cv2
from lax_fit import identify_with_LAX_FIT

# Load images
image1 = cv2.imread('img1.jpg')
image2 = cv2.imread('img2.jpg')

# Calculate LAX-FIT score
score = identify_with_LAX_FIT(image1, image2, visualize=True)
```

### Output Example

```
=== Keypoint Detection ===
Query image keypoints: 1234
Train image keypoints: 1456

=== Matches Length Statistics ===
Mean length: 156.23
Median length: 148.89
Length std dev: 23.45
Length range: 98.76 - 234.56
Number of length measurements: 89

=== Matches Angle Statistics ===
Mean angle: 45.67°
Median angle: 43.21°
Angle std dev: 12.34°
Angle range: 23.45° - 78.90°
Number of angle measurements: 89

=== Final Scores ===
Goodness of Fit Score: 87.65%
LAX-FIT Score: 0.82
```

## Visualization

The algorithm can generate visualizations showing:
- Original feature matches before RANSAC filtering
- Inlier matches after RANSAC filtering
- Match patterns and geometric relationships

## Performance Considerations

- **Computational Complexity**: O(n²) for crossing line analysis
- **Memory Usage**: Linear with number of keypoints
- **Optimization Tips**:
  - Reduce image size for faster processing
  - Adjust minimum inlier threshold based on use case
  - Use visualization only for debugging

## Future Improvements

- Parallel processing for large images
- GPU acceleration for feature detection
- Additional geometric metrics integration
- Machine learning-based score optimization
- Support for different feature detectors

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- New features
- Documentation updates

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

**Roscoe Kerby**
- GitHub: [roscoekerby](https://github.com/roscoekerby)
- LinkedIn: [Roscoe Kerby](https://www.linkedin.com/in/roscoekerby/)

## References

- SIFT: Lowe, D.G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
- RANSAC: Fischler, M.A. & Bolles, R.C. (1981). "Random Sample Consensus"
