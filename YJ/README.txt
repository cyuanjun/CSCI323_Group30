Use the 2 files from FINAL
- (FINAL_image_processing and FINAL_digit_recognition)
- Images are cropped from https://www.sudokuweb.org/ and https://www.websudoku.com/ to ensure reliability.
- Sudoku iamges from other sources may also work but are not as reliable.
- If u want to train yourself, training set used is from https://www.kaggle.com/datasets/kshitijdhama/printed-digits-dataset/versions/56
- Otherwise u can just run with the printed_digits_cnn.keras model.


packages needed:
- tensorflow
- cv2 (opencv-python)
- numpy

Changes to be made
- FINAL_image_processing
  - Line 8 (Change to your image directory)

- FINAL_digit_recognition
  - Line 19 (Change to your dataset directory)
  - Line 20 (Change to the path to the model)
