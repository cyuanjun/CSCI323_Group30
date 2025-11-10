# FT30 Automatic Sudoku Solver 
This project focuses on developing an automated Sudoku solver that integrates image recognition with algorithmic reasoning. The system takes an image of an unsolved Sudoku puzzle, processes it to detect and recognize digits, and applies a backtracking algorithm to generate a valid completed grid.

[Presentation Slides](https://docs.google.com/presentation/d/1hCGh7zgCO9fUspghddKswMbwv7Oy44Nz9_8gRTmTEZE/edit?slide=id.g3a1b57e9357_4_2#slide=id.g3a1b57e9357_4_2)
[Presentation Video]

# Objective
The project aims to create an end-to-end AI system capable of:
  1. Extracting and segmenting Sudoku grids from input images.
  2. Recognizing digits using a trained Convolutional Neural Network (CNN) model.
  3. Solving the puzzle logically through a CSP-based backtracking algorithm.

This pipeline demonstrates how image recognition and search algorithms can work together to solve structured reasoning tasks.

# Getting Started
Built using Python, Numpy, OpenCV, Tensorflow, and Matplotlib.

Environment: Python virtual environment

Algorithm applied: Basic Backtracking, Minimum Remaining Value, Foward Checking, Degree Hueristic, and Least Constraining Value.    

## Clone the Repository
```bash
git clone https://github.com/cyuanjun/CSCI323_Group30.git
cd CSCI323_Group30
```

## Setup Virtual Environment
```bash
python -m venv .venv
# For Windows
venv\Scripts\activate
# For Mac/Linux
source venv/bin/activate
```

## Install Dependencies
(change later, can also include package installer script)
```bash
pip install numpy opencv-python tensorflow matplotlib
```

# Files


