import cv2
import numpy as np

def sort_contours_top_to_bottom(contours, method="top-to-bottom"):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    # sort by y (top) then x (left)
    sorted_pairs = sorted(zip(contours, bounding_boxes), key=lambda b: (b[1][1], b[1][0]))
    return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]

def detect_panels(img, debug=False):
    orig = img.copy()
    h, w = img.shape[:2]

    # Resize for faster processing if very large, keep scale
    max_dim = 1600
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance edges: bilateral or median can help preserve lines
    gray = cv2.medianBlur(gray, 3)

    # Adaptive threshold to pick up dark panel borders on varied backgrounds
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 10)

    # Morphological closing to join border segments, then opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours, keep large rectangular-ish ones
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panel_boxes = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        img_area = img.shape[0] * img.shape[1]
        # Filter by area (ignore tiny blobs) and aspect ratio (avoid thin lines)
        if area < img_area * 0.002:  # tuneable threshold
            continue
        aspect = ww / float(hh + 1e-9)
        # Accept a broad range of aspect ratios; comic panels can be tall or wide
        if aspect < 0.05 or aspect > 20:
            continue
        # Approximate contour to polygon and check rectangularity
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # Either polygon with 4 points or a large blob likely a panel
        if len(approx) == 4 or area > img_area * 0.08:
            panel_boxes.append((x, y, ww, hh))
    merged = merge_boxes(panel_boxes, iou_thresh=0.05)

    # Scale boxes back to original image size if resized
    final_boxes = []
    for (x,y,ww,hh) in merged:
        ix = int(round(x/scale))
        iy = int(round(y/scale))
        iww = int(round(ww/scale))
        ihh = int(round(hh/scale))
        # clamp
        ix = max(0, min(ix, w-1))
        iy = max(0, min(iy, h-1))
        iww = max(1, min(iww, w-ix))
        ihh = max(1, min(ihh, h-iy))
        final_boxes.append((ix, iy, iww, ihh))

    # Sort top-to-bottom, left-to-right
    final_boxes = sorted(final_boxes, key=lambda b: (b[1], b[0]))

    if debug:
        viz = orig.copy()
        for i, (x,y,ww,hh) in enumerate(panel_boxes):
            cv2.rectangle(viz, (x,y), (x+ww, y+hh), (0,128,128), 7)
        for i, (x,y,ww,hh) in enumerate(final_boxes, start=1):
            cv2.rectangle(viz, (x,y), (x+ww, y+hh), (0,255,0), 3)
            cv2.putText(viz, str(i), (x+8, y+24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imwrite("panels_debug.jpg", viz)
        print("Debug visualization saved to panels_debug.jpg")
    
    print("Final Boxes", final_boxes)
    
    return final_boxes

# Merge overlapping boxes (panels sometimes detected in pieces)
def merge_boxes(boxes, iou_thresh=0.15):
    boxes = [list(b) for b in boxes]
    merged = []
    while boxes:
        bx = boxes.pop(0)
        x1,y1,w1,h1 = bx
        xa1, ya1, xa2, ya2 = x1, y1, x1+w1, y1+h1
        changed = False
        i = 0
        while i < len(boxes):
            x2,y2,w2,h2 = boxes[i]
            xb1, yb1, xb2, yb2 = x2, y2, x2+w2, y2+h2
            # intersection
            ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
            ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
            iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
            inter = iw*ih
            union = (xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter
            iou = inter/union if union>0 else 0
            if iou > iou_thresh or (ix1 < ix2 and iy1 < iy2 and (inter > 0)):
                # merge boxes
                nx1 = min(xa1, xb1); ny1 = min(ya1, yb1)
                nx2 = max(xa2, xb2); ny2 = max(ya2, yb2)
                bx = [nx1, ny1, nx2-nx1, ny2-ny1]
                xa1, ya1, xa2, ya2 = nx1, ny1, nx2, ny2
                boxes.pop(i)
                changed = True
            else:
                i += 1
        merged.append(tuple(bx))
    return merged

