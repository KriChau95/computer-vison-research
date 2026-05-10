import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import pycocotools.mask as mask

images_dir = 'saved_images/'
jpgs = os.listdir(images_dir)
coco_dataset_path = 'C:/Users/kc424/COCO/annotations'

for i in range(len(jpgs)):
    jpgs[i] = int(jpgs[i][:-4])



results_dir = 'result_images/'

# instantiate COCO specifying the annotations json path
coco = COCO('annotations\instances_train2017.json')

# Specify a list of category names of interest
catIds = coco.getCatIds(catNms=['person'])

# Get the corresponding image ids and images using loadImgs
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)

# Loading in images if no images present
no_images = len(images) == 0
if no_images:
    print("Loading images for the first time")
    for im in images:
        img_data = requests.get(im['coco_url']).content
        with open('saved_images/' + im['file_name'], 'wb') as handler:
            handler.write(img_data)
else:
    print("Images loaded from before")

""" 
start with image of person (input is img_id) and transform the image into a cropped and padded image containing
the person with original colors, the background as black pixels, and the person is padded by 30 pixels.
"""
def process_image(img_id):
    img = coco.imgs[img_id]
    image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))

    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(anns_ids)

    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])

    masked_image = np.copy(image)
    for i in range(masked_image.shape[0]):
        for j in range(masked_image.shape[1]):
            if int(mask[i][j]) == 2:
                masked_image[i][j] = image[i][j]
            else:
                masked_image[i][j] = [0,0,0]

    cropped_img = rectangular_box(masked_image)
    padded_img = pad_image(cropped_img, 60)

    return padded_img

# Specifying location of a specific test image (322 in this case)
img_dir = 'saved_images/'
img_id = 36
img = coco.imgs[img_id]
image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
original_img = np.copy(image)

plt.figure(figsize = (15,5))

# function that plots an array of images, labeling them with the corresponding title from an array of titles
def plot_images(img_arr, title_arr):
    num_figs = len(img_arr)
    for i in range(num_figs):
        plt.subplot(1, num_figs, i+1)
        plt.imshow(img_arr[i])
        plt.title(title_arr[i])
        plt.axis('off')

# Check loaded annotations
anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(anns_ids)

# Draw annotations on the image
coco.showAnns(anns)

# function that will draw the outline of the person
def draw_person_outline(image, anns):
    for ann in anns:
        segs = ann["segmentation"]

        if len(segs) > 1:  # Multiple polygons
            for seg in segs:
                seg = np.array(seg, np.int32).reshape((1, -1, 2))
                cv2.drawContours(image, [seg], -1, (0, 255, 0), 2)
        else:  # Single polygon
            seg = np.array(segs[0], np.int32).reshape((1, -1, 2))
            cv2.drawContours(image, [seg], -1, (0, 255, 0), 2)

# Iterate through annotations

ann = anns[0]
image_id = ann["image_id"]
image_info = coco.loadImgs(image_id)
image_path = "saved_images/000000000036.jpg"
image = cv2.imread(image_path)
draw_person_outline(image, [ann])  # Draw annotations

# returns image containing green outline of person, and black pixels everywhere else
def extract_outline(image):
    outline_image = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][0] == 0 and image[i][j][1] == 255 and image[i][j][2] == 0:
                outline_image[i][j] = [0,255,0]
            else:
                outline_image[i][j] = [0,0,0]
    return outline_image


outline_image = extract_outline(image)

"""
given an image containing an outline, and a specified thickness, it will return a mask with that corresponding layer of thickness
around the border
"""
def get_surrounding_patch(image, anns, thickness = 14):
    thick_border_image = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][0] == 0 and image[i][j][1] == 255 and image[i][j][2] == 0:
                for di in range(-thickness, thickness + 1):
                    for dj in range(-thickness, thickness + 1):
                        n_i = i + di
                        n_j = j + dj
                        if 0 <= n_i < image.shape[0] and 0 <= n_j < image.shape[1]:
                            thick_border_image[n_i][n_j] = [0,255,0]

    my_mask = coco.annToMask(anns[0])

    # plt.subplot(2, 1, 1)
    # plt.imshow(my_mask)
    # plt.title("my mask")
    # plt.axis('off')
    # plt.show()

    
    for i in range(len(anns)):
        my_mask += coco.annToMask(anns[i])

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if int(my_mask[i][j]) == 2 and thick_border_image[i][j][0] == 0 and thick_border_image[i][j][1] == 0 and thick_border_image[i][j][2] == 0:
                thick_border_image[i][j] = [0,255,0]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if thick_border_image[i][j][0] == 0 and thick_border_image[i][j][1] == 255 and thick_border_image[i][j][2] == 0:
                thick_border_image[i][j] = [255,255,255]

    
    return thick_border_image

thick_border_image = get_surrounding_patch(outline_image, anns)

def overlay_patch(original_img, patch_img):
    result_img = np.copy(patch_img)
    for i in range(patch_img.shape[0]):
        for j in range(patch_img.shape[1]):
            if patch_img[i][j][0] == 255 and patch_img[i][j][1] == 255 and patch_img[i][j][2] == 255:
                result_img[i][j] = original_img[i][j]

    return result_img

almost_final_image = overlay_patch(original_img, thick_border_image)

def all_black_row(img, r):
    for i in range(img.shape[1]):
        if not(img[r][i][0] == 0 and img[r][i][1] == 0 and img[r][i][2] == 0):
            return False
    return True

def all_black_col(img, c):
    for i in range(img.shape[0]):
        if not(img[i][c][0] == 0 and img[i][c][1] == 0 and img[i][c][2] == 0):
            return False
    return True

def rectangular_box(img):
    t_row, b_row, l_col, r_col = 0, img.shape[0] - 1, 0, img.shape[1] - 1
    
    # Find top row
    for i in range(img.shape[0]):
        if not all_black_row(img, i):
            t_row = i
            break
    
    # Find bottom row
    for i in range(img.shape[0] - 1, -1, -1):
        if not all_black_row(img, i):
            b_row = i
            break
    
    # Find left column
    for i in range(img.shape[1]):
        if not all_black_col(img, i):
            l_col = i
            break
    
    # Find right column
    for i in range(img.shape[1] - 1, -1, -1):
        if not all_black_col(img, i):
            r_col = i
            break
    
    cropped_img = np.copy(img[t_row:b_row+1, l_col:r_col+1])
    return cropped_img


cropped_image = rectangular_box(thick_border_image)

def pad_image(img, pad_width = 30):
    height = img.shape[0]
    width = img.shape[1]

    padded_image = np.zeros((height + 2 * pad_width, width + 2 * pad_width, 3), dtype=np.uint8)

    # Copy the original image into the center of the padded array
    padded_image[pad_width:pad_width+height, pad_width:pad_width+width] = img

    return padded_image


final_image = pad_image(cropped_image)

def process_image_2(img_id):

    img_dir = 'saved_images/'

    img = coco.imgs[img_id]
    image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))

    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(anns_ids)

    ann = anns[0]
    image_id = ann["image_id"]
    image_info = coco.loadImgs(image_id)

    num_zeroes = 12- len(str(img_id))

    image_path = "saved_images/" + (num_zeroes) * '0' + str(img_id) + ".jpg"
    image = cv2.imread(image_path)
    draw_person_outline(image, [ann])  # Draw annotations

    outline_image = extract_outline(image)
    
    thick_border_image = get_surrounding_patch(outline_image, anns)
    
    cropped_image = rectangular_box(thick_border_image)

    final_image = pad_image(cropped_image)

    return thick_border_image, final_image



images_to_plot = [original_img, image, thick_border_image, almost_final_image, cropped_image, final_image]
titles = ["Original Image", "IDK", "Thick Border", "Patch", "Cropped", "Final"]

plot_images(images_to_plot, titles)

output_dir = "result_images_3"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_image(image, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Convert image to BGR color space (OpenCV default)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Scale pixel values to the range [0, 255]
    scaled_image = np.clip(bgr_image, 0, 255).astype(np.uint8)
    
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, scaled_image)
    print(f"Image saved successfully as {output_path}")

problem_list = []
save_success = 0

for i in tqdm(range(len(jpgs))):
    try:
        processed_non_cropped, processed_cropped = process_image_2(jpgs[i])
        save_image(processed_non_cropped, "result_non_cropped", f"mask{jpgs[i]}.jpg")
        save_image(processed_cropped, "result_cropped", f"mask_cropped{jpgs[i]}.jpg")
        save_success += 1
        if save_success == 200:
            break
    except:
        problem_list.append(jpgs[i])

exit()

cv2.destroyAllWindows()


    






#--------------------------------------------------------------------------------------------------

plt.show()
exit()
# Plot the image with only the mask (3)
mask = coco.annToMask(anns[0])
for i in range(len(anns)):
    mask += coco.annToMask(anns[i])

plt.subplot(1,5,3)
plt.imshow(mask)
plt.title("Mask by Itself")
plt.axis('off')

# Plot the image with original colors for the person (4)
masked_image = np.copy(image)
for i in range(masked_image.shape[0]):
    for j in range(masked_image.shape[1]):
        if int(mask[i][j]) == 2:
            masked_image[i][j] = image[i][j]
        else:
            masked_image[i][j] = [0,0,0]

img_w_bounding_box = np.copy(masked_image)
for annotation in anns:
    bbox = annotation['bbox']
    bbox = [int(x) for x in bbox]  # Convert bounding box coordinates to integers
    x, y, w, h = bbox
    cv2.rectangle(img_w_bounding_box, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

plt.subplot(1,5,4)
plt.imshow(img_w_bounding_box)
plt.title("Original Colors for Person Pixels")
plt.axis('off')



padded_img = pad_image(cropped_img)

output_dir = "result_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_image(image, filename):
    # Convert image to BGR color space (OpenCV default)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Scale pixel values to the range [0, 255]
    scaled_image = np.clip(bgr_image, 0, 255).astype(np.uint8)
    
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, scaled_image)
    print(f"Image saved successfully as {output_path}")

# Save the padded image
image_322 = process_image(322)
save_image(image_322, f"padded_image_{img_id}.jpg")


problem_list = []
for i in range(len(jpgs)):
    try:
        processed_image = process_image(jpgs[i])
        save_image(processed_image, f"padded_image_{jpgs[i]}.jpg")
    except:
        problem_list.append(jpgs[i])


image_322_gray = cv2.cvtColor(image_322, cv2.COLOR_BGR2GRAY)

plt.subplot(1,5,5)
plt.imshow(image_322_gray)
plt.title("Padded Image")
plt.axis('off')

#Display the plot
plt.show()
