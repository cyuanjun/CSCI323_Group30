# Importing relevant packages
import cv2
import numpy as np



# Config 
img_path = "Images/test22.png"          # <-- CHANGE THIS TO WHERE IMAGES ARE STORED
img_height, img_width = 450, 450



# preprocessing image
def preprocess_image(img_path, debug=True):

    # read image
    img = cv2.imread(img_path)

    # check if image exists
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {img_path}")

    # resize image
    img = cv2.resize(img, (img_width, img_height))

    # convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur image to reduce noise
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)

    # convert image to binary (black / white)
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # invert binary image for contour detection later
    img_invert = cv2.bitwise_not(img_thresh)

    # visualise preprocessing
    if debug:
        def to3(x): 
            return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if len(x.shape)==2 else x

        def label(im, text, color=(0,255,0)):
            """Draws a small label on the top-left of the image."""
            out = im.copy()
            cv2.rectangle(out, (0,0), (150,35), (0,0,0), -1)
            cv2.putText(out, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, color, 2, cv2.LINE_AA)
            return out

        # create labeled tiles
        tiles = [
            label(cv2.resize(img, (220,220)), "Resize"),
            label(cv2.resize(to3(img_gray), (220,220)), "Grayscale"),
            label(cv2.resize(to3(img_blur), (220,220)), "Blur"),
            label(cv2.resize(to3(img_thresh), (220,220)), "Threshold"),
            label(cv2.resize(to3(img_invert), (220,220)), "Inverted"),
        ]

        # stack tiles side-by-side
        vis = np.hstack(tiles)
        cv2.imshow("Preprocessing Steps", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img, img_invert



def find_biggest_contour(img, img_thresh):

    # finding contours
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_area = 0
    biggest_contour = None

    # looping through contours to find biggest contour
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

    # reorder contours
    points = biggest_contour.reshape((4, 2)).astype(np.float32)
    new_points = np.zeros((4, 2), dtype=np.float32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[2] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[3] = points[np.argmax(diff)]

    # visualise contour
    img_contour = img.copy()
    cv2.drawContours(img_contour, [new_points.astype(int)], -1, (0, 255, 0), 3)
    for x, y in points.astype(int):
        cv2.circle(img_contour, (x, y), 8, (0, 0, 255), -1)
    cv2.imshow("Biggest Contour", img_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return new_points



def warp_image(img, corners):

    # define destination coordinates
    destination = np.array([
        [0, 0],
        [img_width - 1, 0],
        [img_width - 1, img_height - 1],
        [0, img_height - 1]
    ], dtype=np.float32)

    # compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners, destination)
    # apply warp transformation
    warped = cv2.warpPerspective(img, matrix, (img_width, img_height))

    # visualisation
    cv2.imshow("warped image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return warped



def split_cells(warped, grid_size=9, gap=8, gap_color=255):

    # measure image and calculate each cell size
    height, width = warped.shape[:2]
    cell_h = height // grid_size
    cell_w = width // grid_size
    is_color = len(warped.shape) == 3

    # visualisation (create blank canvas)
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

    # loop through each row and column and split
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

    # visualisation
    cv2.imshow("Sudoku Cells with Spacing", grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cells



def extract_digits(
    cell,
    border_frac=0.05,        # light edge crop to avoid grid
    line_band_frac=0.12,     # only remove lines near edges
    blank_fg_thresh=0.010,   # blank if too little foreground
    debug=False, win_prefix="SIMPLE", wait=0
):
    
    # (Helper) visualise for debugging
    def to_bgr(x):
        return x if (len(x.shape)==3 and x.shape[2]==3) else cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    # (Helper) 
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
    th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), 1)                     # CHANGED FROM 1, 1=========================================================================

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

    # 5) blank gating
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



# Visualisation
def show_digits_grid(tiles, grid_size=9, gap=6):

    # Ensure tiles exist (use blank white for empty)
    tiles = [t if t is not None else 255 * np.ones((28, 28), np.uint8) for t in tiles]

    th, tw = tiles[0].shape[:2]
    H = grid_size * th + (grid_size + 1) * gap
    W = grid_size * tw + (grid_size + 1) * gap

    # Create white background (so you can see digits clearly)
    canvas = np.full((H, W), 255, dtype=np.uint8)

    # Place each tile with gaps
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            y1 = r * (th + gap) + gap
            x1 = c * (tw + gap) + gap
            canvas[y1:y1+th, x1:x1+tw] = tiles[idx]
            idx += 1

    # Draw Sudoku gridlines (every 3 cells thicker)
    line_color = 0          # black lines
    thin = 1
    thick = 3

    # Vertical lines
    for c in range(1, grid_size):
        x = c * (tw + gap) + gap // 2
        t = thick if c % 3 == 0 else thin
        cv2.line(canvas, (x, 0), (x, H), line_color, t)

    # Horizontal lines
    for r in range(1, grid_size):
        y = r * (th + gap) + gap // 2
        t = thick if r % 3 == 0 else thin
        cv2.line(canvas, (0, y), (W, y), line_color, t)

    cv2.imshow("Sudoku Digits Grid", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





# MAIN
def process_img(img_path=img_path):

    img, img_thresh = preprocess_image(img_path)
    points = find_biggest_contour(img, img_thresh)
    warped_img = warp_image(img, points)
    cells = split_cells(warped_img)
    print(f"Total cells extracted: {len(cells)}")  # should be 81

    output_digits = []
    for i, c in enumerate(cells):
        d = extract_digits(c, debug=False)
        output_digits.append(d)

    digits = [d if d is not None else np.zeros((28, 28), np.uint8) for d in output_digits]
    
    show_digits_grid(digits, gap=10)
    return digits, warped_img



if __name__ == "__main__":
    process_img()