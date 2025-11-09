# Importing relevant packages
import glob
import cv2
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

import FINAL_image_processing as ip



# config
IMG_PATH = "Images/test1.png"          # <-- CHANGE THIS TO WHERE IMAGES ARE STORED
DATASET_DIR = "C:/Users/Jay/Downloads/Programming/Projects/Sudoku/archive/assets"   # <-- CHANGE THIS TO WHERE DATASET IS STORED
MODEL_PATH  = "printed_digits_cnn.keras"                                            # <-- CHANGE THIS IF NEEDED



# dataset loader
def load_printed_digits_dataset(root_dir, invert_if_needed=True):
    """
    Loads a dataset of printed digits from a directory with strict structure:
        root/
          ├── 0/
          ├── 1/
          ├── 2/
          ...
          └── 9/
    Each folder must contain images (PNG or JPG) of the corresponding digit.
    """

    # check if path exists
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    # detect and ensure dataset structure matches what we want
    subdirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
                     key=lambda p: int(p.name))
    
    if len(subdirs) != 10 or any(int(p.name) not in range(10) for p in subdirs):
        raise ValueError(
            f"Error: Dataset must contain exactly 10 digit folders (0–9).\n"
            f"Found: {[p.name for p in subdirs]}"
        )

    # collect image paths
    img_paths = []
    labels = []
    for d in subdirs:
        imgs = glob.glob(str(d / "*.png")) + glob.glob(str(d / "*.jpg"))
        if not imgs:
            print(f"Warning: No images found in folder '{d.name}'")
        img_paths.extend(imgs)
        labels.extend([int(d.name)] * len(imgs))

    if not img_paths:
        raise ValueError("No images found in any digit folder.")

    # --- load and preprocess all images ---
    X, y = [], []
    for p, label in zip(img_paths, labels):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"Skipping unreadable image: {p}")
            continue
        if im.shape != (28, 28):
            im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)

        # match pipeline: WHITE digit on BLACK background
        if invert_if_needed and im.mean() > 127:
            im = cv2.bitwise_not(im)

        X.append(im.astype("float32") / 255.0)
        y.append(label)

    # --- convert to arrays ---
    X = np.array(X, dtype="float32")[..., None]
    y = np.array(y, dtype="int64")

    print(f"Loaded dataset: X={X.shape}, classes={sorted(set(y.tolist()))}")
    return X, y



def split_train_val(X, y, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    cut = int((1 - val_ratio) * len(X))
    tr, va = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[va], y[va]



def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model



import matplotlib.pyplot as plt

def train_printed_digits(DATASET_DIR, MODEL_PATH="printed_digits_cnn.keras",
                         epochs=100, batch_size=128):
    X, y = load_printed_digits_dataset(DATASET_DIR, invert_if_needed=True)
    Xtr, ytr, Xva, yva = split_train_val(X, y, val_ratio=0.2, seed=42)
    model = build_cnn()

    # --- Callbacks ---
    callbacks = [
        # Stop when val_loss hasn't improved for 'patience' epochs
        EarlyStopping(
            monitor="val_loss",
            patience=6,                 # try 4–10; larger datasets may need more
            restore_best_weights=True,  # roll back to the best epoch
            verbose=1
        ),
        # Save the best model (by val_loss) while training
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        # Reduce LR when val_loss plateaus to squeeze a bit more performance
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    history = model.fit(
        Xtr, ytr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(Xva, yva),
        callbacks=callbacks,
        verbose=1
    )

    # --- Learning curves ---
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ModelCheckpoint already saved the best model to MODEL_PATH.
    # We still return both for convenience.
    return model




def predict_board_from_tiles(model, digits, blank_fg_threshold=0.10, conf_thresh=None, debug_blanks=False):

    # prepare tiles as model input (no separate helper function needed)
    tiles = [t if t is not None else np.zeros((28,28), np.uint8) for t in digits]
    X = (np.array(tiles, dtype="float32") / 255.0)[..., None]

    # model prediction
    probs = model.predict(X, verbose=0)
    preds = probs.argmax(axis=1)

    fixed = []
    for idx, (t, p, pvec) in enumerate(zip(tiles, preds, probs)):
        # foreground ratio (white pixels)
        fg = (t > 0).mean()

        # tall/skinny heuristic (signature of "1")
        ys, xs = np.where(t > 0)
        if len(xs) > 0:
            h_span = (ys.max() - ys.min() + 1) / 28.0
            w_span = (xs.max() - xs.min() + 1) / 28.0
            aspect = h_span / max(w_span, 1e-6)
            tall_skinny = (h_span >= 0.55 and w_span <= 0.35) or (aspect >= 2.2 and h_span >= 0.48)
        else:
            h_span = w_span = aspect = 0.0
            tall_skinny = False

        # dynamic blank threshold: looser for tall/skinny tiles
        tile_blank_thresh = min(blank_fg_threshold, 0.05) if tall_skinny else blank_fg_threshold

        # confidence gate
        pmax = float(pvec.max())

        # decide blank or digit
        blank_by_area = (fg < tile_blank_thresh)
        blank_by_conf = (conf_thresh is not None and pmax < conf_thresh and fg < (tile_blank_thresh + 0.03))

        # if tall & skinny, ignore confidence-based blanking (1s often look "uncertain")
        if tall_skinny:
            blank_by_conf = False

        if blank_by_area or blank_by_conf:
            if debug_blanks:
                print(f"[BLANK] idx={idx:02d} fg={fg:.3f} pmax={pmax:.2f} "
                      f"h={h_span:.2f} w={w_span:.2f} asp={aspect:.2f} tall={tall_skinny}")
            fixed.append(0)
        else:
            fixed.append(int(p))

    board = np.array(fixed, dtype=int).reshape(9,9)
    return board, tiles, probs



# visualisation
def draw_board_on_image(warped_bgr, pred_board=None, solved_board=None):
    """
    Overlay predicted (CNN) digits in green and solved digits in red on the same warped image.
    - Green = CNN-predicted digits (already known from photo)
    - Red = Solver-filled digits (new or corrected)
    """
    out = warped_bgr.copy()
    H, W = out.shape[:2]
    ch, cw = H // 9, W // 9

    for r in range(9):
        for c in range(9):
            pos = (c * cw + cw // 3, r * ch + 2 * ch // 3)

            if pred_board is not None:

                pred_val = int(pred_board[r, c])
                # CNN-predicted digits in green
                if pred_val != 0:
                    cv2.putText(out, str(pred_val), pos, cv2.FONT_HERSHEY_SIMPLEX,
                                1.1, (0, 255, 0), 2, cv2.LINE_AA)

            if solved_board is not None:

                solved_val = int(solved_board[r, c])
                # Solver digits (red) — only if blank originally or corrected
                if solved_val != 0 and solved_val != pred_val:
                    pos2 = (pos[0] + cw // 3, pos[1]) if pred_val != 0 else pos
                    cv2.putText(out, str(solved_val), pos2, cv2.FONT_HERSHEY_SIMPLEX,
                                1.1, (255, 0, 0), 2, cv2.LINE_AA)

    return out



# ------------------------------ Main: load/fit model, predict, visualize ------------------------------
def digit_reg(img_path):

    # call image preprocessing (returns extracted tiles + warped image)
    # - digits: list of 81 (28x28 uint8) tiles (white-on-black or blank=0 tile)
    digits, warped_img = ip.process_img(img_path)

    # Load-or-train model
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Loaded trained model:", MODEL_PATH)
    except Exception as e:
        print("No saved model found (or load failed). Training on Printed Digits…")
        model = train_printed_digits(DATASET_DIR, MODEL_PATH, epochs=30, batch_size=128)

    # Use the tiles you already extracted (no re-extraction)
    board, tiles, probs = predict_board_from_tiles(
        model, digits,
        blank_fg_threshold=0.06,   # thin 1s often ~0.03–0.07
        conf_thresh=0.55,
        debug_blanks=False
    )

    print("Predicted board:\n", board)

    vis = draw_board_on_image(warped_img, board)
    cv2.imshow("Recognized Sudoku (Printed Digits CNN)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return board, warped_img


if __name__ == "__main__":
    digit_reg(IMG_PATH)
