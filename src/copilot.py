import os
import zipfile
import rarfile
import cv2
import numpy as np
from ebooklib import epub
import argparse
import logging
import sys
import jb_panels
import pathlib

logging.basicConfig(
    level=logging.DEBUG,  # Set global logging level to DEBUG
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def extract_images(archive_path, temp_dir="temp_pages"):
    os.makedirs(temp_dir, exist_ok=True)
    images = []
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    
    p = pathlib.Path(archive_path) 
    if p.is_dir():
        candidates = p.iterdir()
        for cand in candidates:
            if not cand.is_file():
                continue
            if cand.suffix.lower() in exts:
                images.append(cand)
        temp_pages = p
    elif archive_path.lower().endswith(".cbz"):
        with zipfile.ZipFile(archive_path, "r") as z:
            for name in sorted(z.namelist()):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    out_path = os.path.join(temp_dir, os.path.basename(name))
                    with z.open(name) as f, open(out_path, "wb") as out:
                        out.write(f.read())
                    images.append(out_path)

    elif archive_path.lower().endswith(".cbr"):
        with rarfile.RarFile(archive_path, "r") as r:
            for name in sorted(r.namelist()):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    out_path = os.path.join(temp_dir, os.path.basename(name))
                    with r.open(name) as f, open(out_path, "wb") as out:
                        out.write(f.read())
                    images.append(out_path)

    logger.debug(f"Extracted {len(images)} images")
    return images


def split_and_process(image_path, layout):
    img_base = pathlib.Path(image_path).name.split(".")[0]
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Failed to read image: {image_path}")
        return []

    h, w = img.shape[:2]
    logger.debug(f"Processing {image_path} size={w}x{h}")

    panels = []
    
    if layout is None:
        boxes = [
            (0, 0, w//2, h//2),       # top-left
            (w//2, 0, w, h//2),       # top-right
            (0, h//2, w//2, h),       # bottom-left
            (w//2, h//2, w, h)        # bottom-right
        ]  
    elif layout == "auto":
        debugDir = pathlib.Path("./debug")
        debugDir.mkdir(exist_ok=True)
        
        img_path = pathlib.Path(image_path)
        boxes = jb_panels.detect_panels(img, debugDir / (img_path.stem + "_panels.jpg"))
    else:
        boxes = []
        l = layout.split(";")
        for stx,sty,sbx,sby in l.split(","):
            ftx,fty,fbx,fby = float(stx), float(sty), float(sbx), float(sby)
            tx = int(ftx*w)
            ty = int(fty*h)
            bx = int(fbx*w)
            by = int(fby*h)
            
            boxes.append([tx,ty,bx,by])
            
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        panel = img[y1:y2, x1:x2]
        if panel.size > 0:
        # # Example OpenCV processing
        # panel = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        # panel = cv2.adaptiveThreshold(panel, 255,
        #                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                              cv2.THRESH_BINARY, 11, 2)
        # panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)

            success, buf = cv2.imencode(".jpg", panel)
            if success:
                panels.append(buf.tobytes())
                logger.debug(f"Quadrant {i} encoded")
            else:
                logger.warning(f"Quadrant {i} failed to encode")
        else:
            logger.warning(f"Quadrant {i} has zero size, skipped")
    return panels

def create_epub(archive_path, output_path, args):
    logger.info(f"Creating EPUB: {output_path}")
    book = epub.EpubBook()
    book.set_identifier("comic-quadrants")
    book.set_title(os.path.basename(output_path))
    book.set_language("en")

    images = extract_images(archive_path)
    
    first = args.first
    last = args.last + 1
    if last == 0:
        last = len(images)
        
    spine = []    # ✅ no 'nav' in spine
    toc = []

    for idx, img_path in enumerate(images[first:last], start=1):
        quadrants = split_and_process(img_path, args.layout)
        for q_idx, quad_bytes in enumerate(quadrants, start=1):
            img_id = f"img_{idx}_{q_idx}"
            file_name = f"{img_id}.jpg"
            book.add_item(epub.EpubItem(uid=img_id, file_name=file_name,
                                        media_type="image/jpeg", content=quad_bytes))

            # Well-formed XHTML content
            page = epub.EpubHtml(title=f"Page {idx} Quadrant {q_idx}",
                                 file_name=f"page_{idx}_{q_idx}.xhtml", lang="en")
            page.content = (
                '<!DOCTYPE html>'
                '<html xmlns="http://www.w3.org/1999/xhtml" lang="en">'
                '<head>'
                '<meta charset="utf-8" />'
                '<meta name="viewport" content="width=device-width, initial-scale=1" />'
                '<title>Quadrant</title>'
                '<style>html,body{margin:0;padding:0;background:#000;}'
                'img{display:block;width:100%;height:auto;}</style>'
                '</head>'
                f'<body><img src="{file_name}" alt="Quadrant"/></body>'
                '</html>'
            )
            book.add_item(page)
            spine.append(page)   # ✅ only actual content pages
            toc.append(page)

        logger.debug(f"Page {idx}: {len(quadrants)} quadrants added")

    # Add navigation files, but do NOT include in spine
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    book.spine = spine
    book.toc = toc

    epub.write_epub(output_path, book)
    logger.info(f"✅ EPUB created: {output_path}")

args = None

def main(argv=None):
    global args
    if argv is None:
        argv = sys.argv[1:]
        
    parser = argparse.ArgumentParser(description="Convert CBR/CBZ comics into EPUB with quadrant view.")
    parser.add_argument("comic_file_or_dir", help="Path to the input comic file (.cbr or .cbz) or a directory of page images")
    parser.add_argument("--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument("--first", type=int, default=0, help="First page to extract")
    parser.add_argument("--last", type=int, default=-1, help="Last page to extract, default is end of the book, negative numbers count from end")
    parser.add_argument("--layout", type=str, default="auto", help="Boxes to extract from pages 'tx1,ty1,bx1,by1;tx2,ty2,..' or 'auto'")
    
    args = parser.parse_args(argv)

    input_file = args.comic_file_or_dir
    base_name = os.path.splitext(input_file)[0]
    output_file = base_name + ".epub"

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    create_epub(input_file, output_file, args)


if __name__ == "__main__":
    main(["test_pages", "--first", "20", "--last", "30"])
    #main(["E:/Downloads/LTB 001-585/011_-_Hexenzauber_mit_Micky_und_Goofy_(1._Auflage).cbr", "--first", "11", "--last", "11"])
