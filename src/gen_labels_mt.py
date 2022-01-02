import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


data_root = '/home/lzanella/datasets/MT'
seq_root = '/home/lzanella/datasets/MT/images/train'
label_root = '/home/lzanella/datasets/MT/labels_with_ids/train'
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root) if 8 <= int(s.split('T')[1][:2]) <= 16]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_width = 1600
    seq_height = 1200

    gt_txt = osp.join(data_root, 'gt', seq, 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq)
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, label, _ in gt:
        if mark == 0 or not label == 1:
            continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:04d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)
