"""Stack the published Jung et al. (2021) panels above our reproductions.

    python compare_with_paper.py "<path to Jung 2021 PDF>" "<path to its SI PDF>" [mode]

Needs pymupdf (pip install pymupdf).  Jung et al. is open access (CC BY 4.0);
the generated compare_*.png files contain cropped figures from it, so keep the
attribution if you share them.  They are git-ignored by default.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')

# (pdf index, page, x0, y0, x1, y1) as page fractions, ours
CROPS = {
    'fig2': (0, 3, 0.10, 0.06, 0.92, 0.47, 'fig2_{mode}.png'),
    'fig3': (0, 4, 0.05, 0.065, 0.97, 0.375, 'fig3_{mode}.png'),
    'fig4': (0, 6, 0.08, 0.235, 0.97, 0.715, 'fig4_{mode}.png'),
    'fig5': (0, 7, 0.08, 0.255, 0.65, 0.48, 'fig5_{mode}.png'),
    'si13': (1, 3, 0.20, 0.10, 0.80, 0.80, None),
}


def crop(pdf, page, x0, y0, x1, y1, dpi=200):
    import pymupdf
    doc = pymupdf.open(pdf)
    pg = doc[page - 1]
    r = pg.rect
    clip = pymupdf.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height)
    pix = pg.get_pixmap(dpi=dpi, clip=clip)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]


def main(paper, si, mode):
    pdfs = [paper, si]
    for key, (ip, page, x0, y0, x1, y1, ours) in CROPS.items():
        if ours is None:
            continue
        top = crop(pdfs[ip], page, x0, y0, x1, y1)
        bot = plt.imread(os.path.join(FIG, ours.format(mode=mode)))
        h_top = top.shape[0] / top.shape[1]
        h_bot = bot.shape[0] / bot.shape[1]
        fig = plt.figure(figsize=(12, 12 * (h_top + h_bot) + 0.6))
        a1 = fig.add_axes([0, h_bot / (h_top + h_bot), 1, h_top / (h_top + h_bot)])
        a2 = fig.add_axes([0, 0, 1, h_bot / (h_top + h_bot)])
        a1.imshow(top); a1.axis('off')
        a1.set_title(f'Published: Jung et al., npj Comput. Mater. 7, 122 (2021), {key} (CC BY 4.0)', fontsize=10)
        a2.imshow(bot); a2.axis('off')
        out = os.path.join(FIG, f'compare_{key}_{mode}.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'stepped')
