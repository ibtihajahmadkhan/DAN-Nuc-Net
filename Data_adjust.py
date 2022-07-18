from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import *

def adjust_data(img, mask):
    if np.max(img) > 1:
        img = img / 255.
        mask = mask / 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask

def data_generator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   save_to_dir=None, target_size=(256, 256), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(train_path, classes=[image_folder], class_mode=None,
                                                        color_mode=image_color_mode, target_size=target_size,
                                                        batch_size=batch_size, save_to_dir=save_to_dir,
                                                        save_prefix=image_save_prefix, seed=seed)
    mask_generator = mask_datagen.flow_from_directory(train_path, classes=[mask_folder], class_mode=None,
                                                      color_mode=mask_color_mode, target_size=target_size,
                                                      batch_size=batch_size, save_to_dir=save_to_dir,
                                                      save_prefix=mask_save_prefix, seed=seed)
    train_generator_zip = zip(image_generator, mask_generator)
    for (img, mask) in train_generator_zip:
        img, mask = adjust_data(img, mask)
        yield img, mask
