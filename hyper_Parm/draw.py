import numpy as np
from unoptimized_code import *
from PIL import Image

index_to_vec = {
    1: [-1, 1],
    2: [0, 1],
    3: [1, 1],
    4: [1, 0],
    5: [1, -1],
    6: [0, -1],
    7: [-1, -1],
    8: [-1, 0]
}
def next_pixel(y_pred):
    # y_pred = np.asarray([0.4, 0.3, 0.15, 0.15])
    n, e, s, w = np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])
    some_dir = [n, e, s, w]

    vecs = [out*dir for out, dir, in zip(y_pred, some_dir)]
    vec_sum = sum(vecs)
    all_dir = [(n+w)/np.sqrt(2), n, (n+e)/np.sqrt(2), e, (e+s)/np.sqrt(2), s, (s+w)/np.sqrt(2), w]
    return index_to_vec[np.argmax([np.dot(vec_sum, dir) for dir in all_dir]) + 1]



###implmementation
# my_image = np.zeros((28, 28), dtype=np.float64)
my_image = [[0] * 28] * 28
# current_pixel = (14, 16)
current_pixel=np.array([14,16])
my_image[current_pixel[0]][current_pixel[1]]=1.00
drawn_pixels = [current_pixel]

model = Model.load("model.model")
for i in range(25):
    y_pred = model.predict(np.array(my_image).reshape(-1, 28*28))[0]
    next_pix = next_pixel(y_pred)
    current_pixel += next_pix
    current_pixel[0] = min(27, current_pixel[0])
    current_pixel[1] = min(27, current_pixel[1])
    drawn_pixels.append(current_pixel.copy())
    n = len(drawn_pixels)
    # my_image=my_image.reshape(28,28)
    for j, (r, c) in enumerate(drawn_pixels[:-1]):
        my_image[r][c] = (j + 1) / (2 * n)
        print("value", (j + 1) / (2 * n))
   

    print(f'my image {my_image}')
    a, b = drawn_pixels[-1]
    my_image[a][b] = 1.0
    print("indexes for last pixel", type(a), type(b))
    print(f'last pix {my_image[a][b]}\n')
    
         

img = Image.fromarray(np.uint8(np.array(my_image)*255) , 'L')
img.show()


