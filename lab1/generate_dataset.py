import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random


def get_fonts(folder):
    fonts = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.ttf') or file.endswith('.otf'):
                fonts.append(os.path.join(root, file))
    return fonts


def generate(directory, fonts, num_rotated=3, noise_level=25):
    font_size = 64
    image_size = (128, 128)
    text_color = (0, 0, 0)

    for char in 'ABCDEF':
        char_dir = os.path.join(directory, char)
        os.makedirs(char_dir, exist_ok=True)

        for font_path in fonts:
            image = Image.new('RGBA', image_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_path, font_size)

            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 4)
            draw.text(position, char, font=font, fill=text_color + (255,))
            image_with_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
            image_with_bg.paste(image, (0, 0), image)

            # Сохраняем оригинальное изображение
            font_name = os.path.basename(font_path)  # Имя файла шрифта
            original_filename = f'{char}_{font_name}.png'
            original_path = os.path.join(char_dir, original_filename)
            image_with_bg.save(original_path)

            # Генерация повернутых изображений
            for i in range(num_rotated):
                angle = random.uniform(-30, 30)
                rotated_image = image.rotate(angle, resample=Image.BILINEAR, expand=True)
                rotated_with_bg = Image.new('RGBA', rotated_image.size, (255, 255, 255, 255))
                rotated_with_bg.paste(rotated_image, (0, 0), rotated_image)
                rotated_filename = f'{char}_{font_name}_rotated_{i + 1}.png'
                rotated_path = os.path.join(char_dir, rotated_filename)
                rotated_with_bg.save(rotated_path)

            # Генерация зашумленных изображений
            image_np = np.array(image_with_bg.convert('RGB'))
            noise = np.random.normal(0, noise_level, image_np.shape)
            noisy_image_np = image_np + noise
            noisy_image_np = np.clip(noisy_image_np, 0, 255).astype(np.uint8)
            noisy_image = Image.fromarray(noisy_image_np)
            noisy_filename = f'{char}_{font_name}_noisy.png'
            noisy_path = os.path.join(char_dir, noisy_filename)
            noisy_image.save(noisy_path)


if __name__ == "__main__":
    fonts_learn_dir = 'fonts/learning'
    fonts_test_dir = 'fonts/testing'

    fonts_learn = get_fonts(fonts_learn_dir)
    fonts_test = get_fonts(fonts_test_dir)

    generate('learning_dataset', fonts_learn)
    generate('testing_dataset', fonts_test)
