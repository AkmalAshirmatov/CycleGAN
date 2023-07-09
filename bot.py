from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command, Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

import logging
import numpy as np
import torch
from PIL import Image
import io
import torchvision.transforms as transforms

import options.options_bot as opt
import models.cycle_gan_model
import utils.additional

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# creating model
model = models.cycle_gan_model.CycleGANModel(opt)

API_TOKEN = opt.BOT_TOKEN

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())

def get_transform(opt, method=transforms.InterpolationMode.BICUBIC):
    osize = [opt.load_size, opt.load_size]
    transform_list = [transforms.Resize(osize, method), transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class Form(StatesGroup):
    waiting_for_style = State()

@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    await message.reply("Welcome! Send me a picture and tell me whether you want it to be converted to winter or summer.")

@dp.message_handler(content_types=types.ContentType.PHOTO, state='*')
async def convert_image(message: types.Message, state: FSMContext):
    # Download image file
    file_id = message.photo[-1].file_id  # photo is stored in different sizes, we take the largest one
    file_info = await bot.get_file(file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    
    # Open image with Pillow
    image = Image.open(io.BytesIO(downloaded_file.getvalue()))

    # Save image in state
    await state.update_data(image=image)

    # Ask user to specify the conversion style
    await Form.waiting_for_style.set()

    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("Summer", callback_data="summer"))
    keyboard.add(InlineKeyboardButton("Winter", callback_data="winter"))
    
    await message.reply("What style do you want?", reply_markup=keyboard)

@dp.callback_query_handler(state=Form.waiting_for_style)
async def choose_style(callback_query: types.CallbackQuery, state: FSMContext):
    # Fetch image from state
    user_data = await state.get_data()
    image = user_data['image']
    h, w = image.size
    image = get_transform(opt)(image)

    # Convert using your models
    if callback_query.data == 'summer':
        with torch.no_grad():
            converted_image = model.netG_B(image)
        #converted_image = convert_to_summer(image)  # replace with your function
    elif callback_query.data == 'winter':
        with torch.no_grad():
            converted_image = model.netG_A(image)
        #converted_image = convert_to_winter(image)  # replace with your function

    converted_image = utils.additional.tensor2im(converted_image)
    converted_image = Image.fromarray(converted_image)
    converted_image = converted_image.resize((h, w), Image.BICUBIC)
    #utils.additional.save_image(generated_image, save_file_name_out, h, w)

    # Convert back to bytes
    byte_arr = io.BytesIO()
    converted_image.save(byte_arr, format='PNG')
    byte_arr.seek(0)

    # Send converted image
    await bot.send_photo(callback_query.from_user.id, photo=byte_arr, caption="Here's your converted image.")

    # Reset state
    await state.finish()

if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)
