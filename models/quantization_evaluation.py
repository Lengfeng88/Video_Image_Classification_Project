import time
import cv2

# Test function
def test_inference_speed(model_path, prototxt_path, test_image):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # Warm up
    for _ in range(10):
        blob = cv2.dnn.blobFromImage(test_image, 1.0, (227, 227), (0, 0, 0), swapRB=False)
        net.setInput(blob)
        net.forward()
    
    # Measure
    start = time.time()
    for _ in range(100):
        blob = cv2.dnn.blobFromImage(test_image, 1.0, (227, 227), (0, 0, 0), swapRB=False)
        net.setInput(blob)
        net.forward()
    end = time.time()
    
    return 100 / (end - start)  # FPS

# Load a sample image for testing
test_img = cv2.imread('image.jpg')

# Test with quantized model (rename .quant file temporarily to disable)
fps_quantized = test_inference_speed('models/age_net.caffemodel', 'models/age_deploy.prototxt', test_img)
print(f"INT8 FPS: {fps_quantized:.2f}")

# Test without quantization (rename .quant file to .quant.bak)
fps_fp32 = test_inference_speed('models/age_net.caffemodel', 'models/age_deploy.prototxt', test_img)
print(f"FP32 FPS: {fps_fp32:.2f}")

print(f"Speedup: {fps_quantized/fps_fp32:.2f}x")