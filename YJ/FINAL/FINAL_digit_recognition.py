# Importing relevant packages
import glob
import cv2
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
import FINAL_Image_processing as ip



# call image preprocessing (returns extracted tiles + warped image)
# - digits: list of 81 (28x28 uint8) tiles (white-on-black or blank=0 tile)
digits, warped_img = ip.process_img()



# config
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



def train_printed_digits(DATASET_DIR, MODEL_PATH="printed_digits_cnn.keras",
                         epochs=10, batch_size=128):
    X, y = load_printed_digits_dataset(DATASET_DIR, invert_if_needed=True)
    Xtr, ytr, Xva, yva = split_train_val(X, y, val_ratio=0.2, seed=42)
    model = build_cnn()
    model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size,
              validation_data=(Xva, yva))
    model.save(MODEL_PATH)
    print("Saved model to:", MODEL_PATH)
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
def draw_board_on_image(warped_bgr, board):
    out = warped_bgr.copy()
    H, W = out.shape[:2]
    ch, cw = H//9, W//9
    for r in range(9):
        for c in range(9):
            v = int(board[r,c])
            if v == 0: 
                continue
            pos = (c*cw + cw//3, r*ch + 2*ch//3)
            cv2.putText(out, str(v), pos, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2, cv2.LINE_AA)
    return out


# ------------------------------ Main: load/fit model, predict, visualize ------------------------------
def digit_reg():
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

    return board


if __name__ == "__main__":
    digit_reg()
