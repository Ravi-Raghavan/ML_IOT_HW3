import numpy as np

def convolution2D(image, kernel):
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate the output dimensions
    output_height = image_height - kernel_height + 1
    #TODO: Compute the output height
    output_width = image_width - kernel_width + 1
    #TODO: Compute the output width
    
    # Initialize the output feature map
    output = np.zeros((output_height, output_width))
    
    # Perform 2D convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest (ROI) from the image
            #todo: Extract the region of interest (ROI) from the image
            
            ROI = image[i:i+kernel_height, j:j+kernel_width]
            
            # Compute the element-wise multiplication and sum
            #todo: Compute the element-wise multiplication and sum
            output[i][j] = np.sum(ROI * kernel)
    
    return output

# Example usage:
if __name__ == "__main__":
    # Create a sample grayscale image (8x8 pixels)
    image = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                      [8, 7, 6, 5, 4, 3, 2, 1],
                      [1, 2, 3, 4, 5, 6, 7, 8],
                      [8, 7, 6, 5, 4, 3, 2, 1],
                      [1, 2, 3, 4, 5, 6, 7, 8],
                      [8, 7, 6, 5, 4, 3, 2, 1],
                      [1, 2, 3, 4, 5, 6, 7, 8],
                      [8, 7, 6, 5, 4, 3, 2, 1]])
    
    # Create a sample kernel (3x3)
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])
    
    # Perform 2D convolution
    result = convolution2D(image, kernel)
    print(result)
