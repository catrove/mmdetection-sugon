import mmdet
from mmdet.datasets import Obj365Dataset
from draw import getRep
from collections import defaultdict, Counter

data_root = '/public/software/apps/DeepLearning/Data/objects365/'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)

cornernet_mode = True
d = Obj365Dataset(
        ann_file=data_root + 'objects365_train.json',
        img_prefix=data_root + 'train/',
        img_scale=(511, 511),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=1,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        cornernet_mode=cornernet_mode,
        resize_keep_ratio=False)

print(d.cat_ids)
exit(0)

def matchBox(box1,box2):
    for i in range(4):
        if abs(box1[i]-box2[i]) > 30:
            return False
    return True

rep, img_id = getRep()
table = [defaultdict(int) for _ in range(366)]
for i in range(len(d)):
    ann_info = (d.get_ann_info(i))
    gt_bboxes = ann_info['bboxes']
    
    gt_labels = ann_info['labels']
    for j in range(len(rep[img_id[i]])):
        _score , _cls , _box = rep[img_id[i]][j]
        for k in range(len(gt_bboxes)):
            match = matchBox(gt_bboxes[k],_box)
            if match:
                table[_cls][gt_labels[k]] += 1

for i in range(len(table)):
    if table[i]:
        try :
            table[i] = list(Counter(table[i]).most_common(1))[0][0]
        except:
            table[i] = -1
    else :
        table[i] = -1
    
        

print(table)
            
