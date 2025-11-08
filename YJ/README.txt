WORKING.py
- Working now from tested images
- Images are cropped from https://www.sudokuweb.org/ to ensure similar fonts.
- Sudoku iamges from other sources may also work but are not as reliable.
- If u want to train yourself, training set used is from https://www.kaggle.com/datasets/kshitijdhama/printed-digits-dataset/versions/56
- Otherwise u can just run with the printed_digits_cnn.keras model.


packages needed:
- tensorflow
- cv2 (opencv-python)
- numpy
- glob
- pathlib

paths
- line 8 (input image path)
- line 292 (dataset directory)
- line293 (model path)
