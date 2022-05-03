import  os

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='')
    parser.add_argument('--list_name', type=str, default=None,
                        help='')
    parser.add_argument('--write_name', type=str, default=None,
                        help='')
    args = parser.parse_args()
    return args

def generate_frame_list(write_name, dir_name, list_name):
    with open(write_name, 'w') as f, open(os.path.join(dir_name, list_name), 'r') as g:
        for line in g.readlines():
            gt_dir, mask_dir, tran_dir = line.split()
            gt_dir = gt_dir.replace('\\', '/')
            mask_dir = mask_dir.replace('\\', '/')
            tran_dir = tran_dir.replace('\\', '/')
            gts = os.listdir(os.path.join(dir_name, gt_dir))
            masks = os.listdir(os.path.join(dir_name, mask_dir))
            trans = os.listdir(os.path.join(dir_name, tran_dir))
            gts.sort()
            masks.sort()
            trans.sort()
            for i in range(len(masks)):
                gt_name = os.path.join(gt_dir, trans[i])
                masks_name = os.path.join(mask_dir, masks[i])
                tran_name = os.path.join(tran_dir, trans[i])
                assert os.path.exists(os.path.join(dir_name, gt_name))
                assert gt_name[-9:-4] == masks_name[-9:-4]
                assert gt_name[-9:-4] == tran_name[-9:-4]
                f.write(gt_name + ' ' + masks_name + ' ' + tran_name + '\n')
args = parse_args()
generate_frame_list(args.write_name, args.dataset_path, args.list_name)
#generate_frame_list('test_frames.txt', args.dataset_path, 'test_list.txt')




