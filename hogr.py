from enum import Enum
import os.path
import cv2
import numpy as np

# Define Test Images
def test_images():
   
    test_directory = './NICTA/TestSet/PositiveSamples'
    test_set = [
        'item_00000010.pnm'
    ]
    for image, _ in enumerate(test_set):
        yield os.path.join(test_directory, test_set[image])


def gamma_correction(image, gamma=1.0):
    
    bits_pixel = np.iinfo(image.dtype).max

    normal_pic = image / np.max(image)
    new_pic = bits_pixel * np.power(normal_pic, gamma)
    new_pic = new_pic.astype(np.uint8)

    return new_pic


def compute_gradients(image, is_signed=False):
   
    kernel_a = np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]])
    gradient_x = cv2.filter2D(image, -1, kernel_a)
    gradient_y = cv2.filter2D(image, -1, np.transpose(kernel_a))

    orientation_size = 360 if is_signed else 180

    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    angle = np.arctan2(gradient_y, gradient_x) * (orientation_size / np.pi)
    return gradient_x, gradient_y, magnitude, angle


def calculate_histogram(magnitudes, angles, bin_count=12, is_signed=False):
  
    cell_size_x, cell_size_y = magnitudes.shape
    orientation_size = 360 if is_signed else 180
    bin_width = orientation_size // bin_count
    cell_histogram = np.zeros(bin_count)
    for row in range(cell_size_x):
        for col in range(cell_size_y):
            orientation = angles[row][col]
            histogram_bin = int(orientation // bin_width)
            cell_histogram[histogram_bin] += magnitudes[row][col]
    return cell_histogram / np.product(magnitudes.shape)


def compute_weighted_vote(gradient, cell_size=(8, 8), bin_count=12, is_signed=False):
    

    gradient_x = gradient[0]
    gradient_y = gradient[1]
    gradient_magnitudes = gradient[2]
    gradient_angles = gradient[3]

    grad_size_x, grad_size_y = gradient_magnitudes.shape
    cell_size_x, cell_size_y = cell_size
    cell_count_x = int(grad_size_x / cell_size_x)  # Number of cells in x axis
    cell_count_y = int(grad_size_y / cell_size_y)  # Number of cells in y axis

    #print("[INFO] Cell counts:  x={} y={}".format(cell_count_x, cell_count_y))
    hog_cells = np.zeros((cell_count_x, cell_count_y, bin_count))

    prev_x = 0
    # Compute HOG of each cell
    for row in range(cell_count_x):
        prev_y = 0
        for col in range(cell_count_y):
            magnitudes_cell = gradient_magnitudes[prev_x:prev_x + cell_size_x, prev_y:prev_y + cell_size_y]
            angles_cell = gradient_angles[prev_x:prev_x + cell_size_x, prev_y:prev_y + cell_size_y]
            hog_cells[row][col] = calculate_histogram(magnitudes_cell, angles_cell, bin_count, is_signed)

            prev_y += cell_size_y
        prev_x += cell_size_x

    #print("[DEBUG] Cells array shape:    {}".format(hog_cells.shape))

    return hog_cells, (cell_count_x, cell_count_y)


def contrast_normalize(vector, epsilon=1e-5):
   
    #print("[DEBUG]  What am I normalizing?:  {}".format(vector.shape))
    return vector / np.sqrt(np.linalg.norm(np.square(vector), 2) + np.square(epsilon))


def normalize_blocks(cells, cell_size=(8, 8), block_size=(16, 16), bin_count=12):
    
    cell_size_x, cell_size_y = cells.shape[:2]
    block_size_x, block_size_y = block_size
    block_count_x = cell_size_x - 1
    block_count_y = cell_size_y - 1
    cells_per_block_x = int(block_size_x // cell_size[0])
    cells_per_block_y = int(block_size_y // cell_size[1])
    #print("[INFO] Block counts:  x={} y={}".format(block_count_x, block_count_y))

    normalized_blocks = np.zeros((block_count_x, block_count_y, cells_per_block_x*cells_per_block_y*bin_count))

    # Normalize HOG by block
    for row in range(block_count_x):
        for col in range(block_count_y):
            xrange = row+cells_per_block_x
            yrange = col+cells_per_block_y
            #print("[DEBUG] Row={} Col={}\n\t Getting cells {} and {}".format(row, col, xrange, yrange))
            hog_block = cells[row:row + cells_per_block_x, col:col + cells_per_block_y].ravel()
            normalized_blocks[row, col] = contrast_normalize(hog_block)

    return normalized_blocks, (block_count_x, block_count_y)


if __name__ == '__main__':
    for my_image in test_images():
        # Step 1 - Input Image
        # Load in the test image, resize to 64x128, and convert to grayscale
        print("[INFO] Loading test image {}".format(my_image))
        test_image = cv2.imread(my_image)
        test_image = cv2.resize(test_image, (64, 128))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

        # Step 2 - Normalize gamma and color
        gamma_value = 1.0
        test_image = gamma_correction(test_image, gamma_value)

        # Step 3 - Compute gradients
        test_gradient = compute_gradients(test_image)
        TEST_GRADIENT = True
        if TEST_GRADIENT:
            gx, gy = test_gradient[0], test_gradient[1]
            gheight = gx.shape[1] * 4
            gwidth = gx.shape[0] * 4
            gx = cv2.resize(gx, (gheight, gwidth))
            gy = cv2.resize(gy, (gheight, gwidth))
            output_stack = np.hstack((gx, gy))
            cv2.imshow('Filter results', output_stack)
            cv2.waitKey(0)

        # Step 4 - Weighted vote into spatial and orientation cells
        cell_histograms, _ = compute_weighted_vote(test_gradient)

        # Step 5 - Contrast normalize over overlapping spatial blocks
        hog_blocks, _ = normalize_blocks(cell_histograms)
        # Step 6 - Collect HOG's over detection window

        print(hog_blocks.ravel().shape)
        # Step 7 - Linear SVM

    cv2.destroyAllWindows()

