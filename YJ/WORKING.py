# Importing relevant packages
import cv2
import numpy as np



# ---------- Configuration ----------
img_path = "Images/image5.png"
img_height, img_width = 450, 450



# preprocessing image
def preprocess_image(img_path):

    # read image
    img = cv2.imread(img_path)

    cv2.imshow("Input image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # check if image can be found
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {img_path}")

    # resize image
    img = cv2.resize(img, (img_width, img_height))

    cv2.imshow("Input image (resized)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    img_thresh = cv2.bitwise_not(img_thresh)

    cv2.imshow("Input image (thresh)", img_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img, img_thresh



def find_biggest_contour(img, img_thresh):
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_area = 0
    biggest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > biggest_area and len(approx) == 4:
                biggest_area = area
                biggest_contour = approx
    if biggest_contour is None:
        raise Exception("No suitable Sudoku contour found!")

    points = biggest_contour.reshape((4, 2)).astype(np.float32)
    new_points = np.zeros((4, 2), dtype=np.float32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[2] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[3] = points[np.argmax(diff)]

    img_contour = img.copy()
    cv2.drawContours(img_contour, [new_points.astype(int)], -1, (0, 255, 0), 3)
    for x, y in points.astype(int):
        cv2.circle(img_contour, (x, y), 8, (0, 0, 255), -1)
    cv2.imshow("Biggest Contour", img_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return new_points



def warp_image(img, corners):
    destination = np.array([
        [0, 0],
        [img_width - 1, 0],
        [img_width - 1, img_height - 1],
        [0, img_height - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(img, matrix, (img_width, img_height))

    cv2.imshow("warped image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return warped



def split_cells(warped, grid_size=9, gap=8, gap_color=255):

    height, width = warped.shape[:2]
    cell_h = height // grid_size
    cell_w = width // grid_size
    is_color = len(warped.shape) == 3

    # background (white) canvas large enough for 9 cells + gaps
    if is_color:
        grid_img = np.full(
            (height + gap * (grid_size + 1),
             width + gap * (grid_size + 1), 3),
            gap_color, dtype=np.uint8
        )
    else:
        grid_img = np.full(
            (height + gap * (grid_size + 1),
             width + gap * (grid_size + 1)),
            gap_color, dtype=np.uint8
        )

    cells = []
    for r in range(grid_size):
        for c in range(grid_size):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            cell = warped[y1:y2, x1:x2]
            cells.append(cell)

            # compute placement with gap
            gy1 = r * (cell_h + gap) + gap
            gx1 = c * (cell_w + gap) + gap
            grid_img[gy1:gy1 + cell_h, gx1:gx1 + cell_w] = cell

    cv2.imshow("Sudoku Cells with Spacing", grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cells






def extract_digit_simple(
    cell,
    border_frac=0.05,        # light edge crop to avoid grid
    line_band_frac=0.12,     # only remove lines near edges
    blank_fg_thresh=0.010,   # blank if too little foreground
    debug=False, win_prefix="SIMPLE", wait=0
):
    # --- helpers ---
    def to_bgr(x): return x if (len(x.shape)==3 and x.shape[2]==3) else cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    def tile(imgs, cols=3, scale=2):
        h0, w0 = imgs[0].shape[:2]
        imgs = [cv2.resize(i, (w0, h0), interpolation=cv2.INTER_NEAREST) for i in imgs]
        rows = [cv2.hconcat([to_bgr(x) for x in imgs[r:r+cols]]) for r in range(0, len(imgs), cols)]
        grid = cv2.vconcat(rows)
        if scale != 1:
            H, W = grid.shape[:2]
            grid = cv2.resize(grid, (W*scale, H*scale), interpolation=cv2.INTER_NEAREST)
        return grid

    # 1) grayscale + light border crop
    gray0 = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape)==3 else cell
    H0, W0 = gray0.shape[:2]
    b = max(1, int(border_frac * min(H0, W0)))
    gray = gray0[b:H0-b, b:W0-b] if (H0>2*b and W0>2*b) else gray0

    # 2) mild blur + adaptive threshold (digit = white, background black)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # GAUSSIAN works better than MEAN here
        cv2.THRESH_BINARY_INV,
        31, 3   # try blockSize 31 (or 25/35) and C around 2–5
    )


    # 3) gentle speck removal
    th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((1,1), np.uint8), 1)

    # 4) remove long grid lines ONLY near borders
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    horiz = cv2.morphologyEx(th_clean, cv2.MORPH_OPEN, k_h, 1)
    vert  = cv2.morphologyEx(th_clean, cv2.MORPH_OPEN, k_v, 1)
    grid_mask_full = cv2.bitwise_or(horiz, vert)

    H, W = th_clean.shape[:2]
    band = max(1, int(line_band_frac * min(H, W)))
    border_band = np.zeros((H, W), np.uint8)
    border_band[:band,:] = 255; border_band[-band:,:] = 255
    border_band[:,:band] = 255; border_band[:,-band:] = 255

    grid_mask = cv2.bitwise_and(grid_mask_full, border_band)
    digit_only = cv2.bitwise_and(th_clean, cv2.bitwise_not(grid_mask))

    # 5) blank gating (very simple + robust)
    if (digit_only > 0).mean() < blank_fg_thresh:
        if debug:
            dbg = tile([gray, th_clean, grid_mask, digit_only, np.zeros((digit_only.shape), np.uint8)], cols=5, scale=2)
            cv2.imshow(f"{win_prefix} – blank", dbg); cv2.waitKey(wait); cv2.destroyAllWindows()
        return None

    # 6) direct resize to 28×28 (keep raw shape; no closing/dilation)
    out28 = cv2.resize(digit_only, (28, 28), interpolation=cv2.INTER_AREA)

    if debug:
        dbg = tile([to_bgr(gray), to_bgr(th_clean), to_bgr(grid_mask),
                    to_bgr(digit_only), to_bgr(out28)], cols=5, scale=2)
        cv2.imshow(f"{win_prefix}", dbg); cv2.waitKey(wait); cv2.destroyAllWindows()

    return out28







# ---------- Addition: visualize all extracted digits ----------
def show_all_digits(cells, grid_size=9, scale=2):
    """
    Runs extract_digit on each cell and shows a 9x9 collage in one OpenCV window.
    Empty cells are shown as white 28x28 tiles. Also prints a simple report.
    """
    processed = []
    empties = []
    for i, cell in enumerate(cells):
        d = extract_digit_simple(cell)
        if d is None:
            d = 255 * np.ones((28, 28), dtype=np.uint8)  # blank tile
            empties.append(i)
        processed.append(d)

    # stitch 9x9 grayscale tiles
    rows = []
    for r in range(grid_size):
        row_tiles = processed[r*grid_size:(r+1)*grid_size]
        rows.append(cv2.hconcat(row_tiles))
    grid = cv2.vconcat(rows)

    # scale for visibility
    if scale != 1:
        h, w = grid.shape[:2]
        grid = cv2.resize(grid, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("All Extracted Digits (28x28 each)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # console summary
    print(f"Extracted digits collage shown. Empty cells: {len(empties)}")
    if empties:
        print("Empty cell indices:", empties)
# --------------------------------------------------------------



img, img_thresh = preprocess_image(img_path)
points = find_biggest_contour(img, img_thresh)
warped_img = warp_image(img, points)
cells = split_cells(warped_img)
print(f"Total cells extracted: {len(cells)}")  # should be 81

for i, c in enumerate(cells):
    print("Cell", i)
    _ = extract_digit_simple(c, debug=False, win_prefix=f"Cell {i}", wait=0)


show_all_digits(cells, grid_size=9, scale=2)







# ========================= CNN TRAIN + PREDICT (APPEND-ONLY) =========================
# Set this to your local dataset path (folder with images).
# Supported layouts:
#   A) printed_digits_dataset/0/*.png ... printed_digits_dataset/9/*.png
#   B) printed_digits_dataset/*.png  (filename contains the label, e.g., "5_123.png")
DATASET_DIR = "C:/Users/Jay/Downloads/Programming/Projects/Sudoku/archive/assets"   # <-- CHANGE THIS
MODEL_PATH  = "printed_digits_cnn.keras"

import os, glob
from pathlib import Path
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ---- Loader: works for both folder-per-class and flat files ----
def load_printed_digits_dataset(root_dir, invert_if_needed=True):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    subdirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    if subdirs:
        # folder-per-class
        img_paths = []
        for d in sorted(subdirs, key=lambda p: int(p.name)):
            img_paths += glob.glob(str(d / "*.png")) + glob.glob(str(d / "*.jpg"))
        def get_label(p): return int(Path(p).parent.name)
    else:
        # flat files with digit in the filename
        img_paths = glob.glob(str(root / "*.png")) + glob.glob(str(root / "*.jpg"))
        def get_label(p):
            name = Path(p).stem.replace("-", "_")
            for tok in name.split("_"):
                if tok.isdigit() and len(tok) == 1:
                    return int(tok)
            if name and name[0].isdigit():
                return int(name[0])
            raise ValueError(f"Cannot parse label from filename: {p}")

    X, y = [], []
    for p in img_paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if im.shape != (28, 28):
            im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
        # match pipeline: WHITE digit on BLACK background
        if invert_if_needed and im.mean() > 127:  # white background -> invert
            im = cv2.bitwise_not(im)
        X.append(im.astype("float32")/255.0)
        y.append(get_label(p))
    X = np.array(X, dtype="float32")[..., None]
    y = np.array(y, dtype="int64")
    print(f"Loaded dataset: X={X.shape}, classes={sorted(set(y.tolist()))}")
    return X, y

# ---- Simple train/val split ----
def split_train_val(X, y, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    cut = int((1 - val_ratio) * len(X))
    tr, va = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[va], y[va]

# ---- CNN model ----
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
    print("✅ Saved model to:", MODEL_PATH)
    return model

# ---- Convert your cells -> tiles -> model input ----
def tiles_from_cells_using_simple_extractor(cells):
    tiles = []
    for cell in cells:
        t = extract_digit_simple(cell)   # uses your existing function
        if t is None:
            t = np.zeros((28,28), np.uint8)  # blank tile
        tiles.append(t)
    X = (np.array(tiles)/255.0).astype("float32")[..., None]
    return tiles, X

# ---- Predict board with blank gating ----
def predict_board_with_model(model, cells, blank_fg_threshold=0.10, conf_thresh=None, debug_blanks=False):
    """
    blank_fg_threshold: base foreground % gate on the 28x28 tile
    conf_thresh: optional softmax confidence gate (e.g., 0.55)
    debug_blanks: print why a tile was blanked
    """
    tiles, X = tiles_from_cells_using_simple_extractor(cells)
    probs = model.predict(X, verbose=0)
    preds = probs.argmax(axis=1)

    fixed = []
    for idx, (t, p, pvec) in enumerate(zip(tiles, preds, probs)):
        # --- foreground ratio ---
        fg = (t > 0).mean()

        # --- tall/skinny heuristic (signature of "1") ---
        ys, xs = np.where(t > 0)
        if len(xs) > 0:
            h_span = (ys.max() - ys.min() + 1) / 28.0
            w_span = (xs.max() - xs.min() + 1) / 28.0
            aspect = h_span / max(w_span, 1e-6)
            tall_skinny = (h_span >= 0.55 and w_span <= 0.35) or (aspect >= 2.2 and h_span >= 0.48)
        else:
            h_span = w_span = aspect = 0.0
            tall_skinny = False

        # --- dynamic blank threshold: looser for tall/skinny tiles ---
        tile_blank_thresh = min(blank_fg_threshold, 0.05) if tall_skinny else blank_fg_threshold

        # --- confidence gate ---
        pmax = float(pvec.max())

        # --- decide blank or digit ---
        blank_by_area = (fg < tile_blank_thresh)
        blank_by_conf = (conf_thresh is not None and pmax < conf_thresh and fg < (tile_blank_thresh + 0.03))

        # If it's tall & skinny, ignore the confidence-based blanking (1s often look "uncertain")
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


# ---- Draw predictions onto your warped image ----
def draw_board_on_image(warped_bgr, board):
    out = warped_bgr.copy()
    H, W = out.shape[:2]
    ch, cw = H//9, W//9
    for r in range(9):
        for c in range(9):
            v = int(board[r,c])
            if v == 0: continue
            pos = (c*cw + cw//3, r*ch + 2*ch//3)
            cv2.putText(out, str(v), pos, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2, cv2.LINE_AA)
    return out

# ---- Load-or-train, then predict on your current image ----
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Loaded trained model:", MODEL_PATH)
except Exception as e:
    print("ℹ️ No saved model found (or load failed). Training on Printed Digits…")
    model = train_printed_digits(DATASET_DIR, MODEL_PATH, epochs=30, batch_size=128)

board, tiles, probs = predict_board_with_model(
    model, cells,
    blank_fg_threshold=0.06,   # was 0.10; thin 1s often ~0.03–0.07
    conf_thresh=0.55,
    debug_blanks=False         # set True once to see why cells blank out
)

print("Predicted board:\n", board)

vis = draw_board_on_image(warped_img, board)
cv2.imshow("Recognized Sudoku (Printed Digits CNN)", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ====================================================================================


