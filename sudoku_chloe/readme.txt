# virtual environment setup
python -m venv venv

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

+ need to install tesseract engine for OCR :
https://github.com/UB-Mannheim/tesseract/wiki -> download here
and copy paste the path to the environment variables in your local computer
and use this command to check whether tesseract is successfully installed : tesseract --version

pip install opencv-python numpy tensorflow keras pytesseract matplotlib

^^ need to use this command to install necessary modules

update : 08/11/2025
i didnt use mnist handwritten database since the image i input is not handwritten.
so i used generate_sudoku_digits.py to generate test_images folder that contains all kinds of numbers in different fonts.
After generating sudoku_digits, you can go to the train_model_without_mnist.py and run. it will generate digit_model_sudoku.h5 which will be used in digit_recognition.py!
ok now ur setup is finished, try to run main.py to see if it work s.
to change the input image, check line 29 of main.py
image_path = "test_images/image3.png"
^^ can change this name to use different image
