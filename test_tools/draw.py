import cv2
import sys
import json
import heapq
from tf import cat2real

def replace():
    jname = open('/public/home/fengshen/private/CentripetalNet/mmdetection/Object365-x32/work_dirs/gloo_32x4_centripetalnet_mask_hg104/temp_0.json','r')
    stats = json.load(jname)
    for i in stats:
        cls = i['category_id']
        if cat2real[cls] != -1:
            i['category_id'] = cat2real[cls]
    json.dump(stats,open('temp_rep.json','w'))

        

def getRep():
    jname = open('/public/home/fengshen/private/CentripetalNet/mmdetection/Object365-x32/work_dirs/gloo_32x4_centripetalnet_mask_hg104/temp_0.json','r')
    rep = dict()
    print("loading json ...")
    stats = json.load(jname)
    print("finish loading json")
    count = 0
    index = []
    for i in stats:
        #if count > 5000:
        #    break
        img_id = i['image_id']
        score = i['score']
        cls = i['category_id']
        x,y,w,h = i['bbox']
        box = [x,y,x+w,y+h]
        if not img_id in rep:
            count += 1
            rep[img_id] = [[score,cls,box]]
            index.append(img_id)
        #elif len(rep[img_id]) < 30:
        else:
            rep[img_id].append([score, cls, box])
    for i in index:
        rep[i] = list(heapq.nlargest(15,rep[i]))
    return rep,index

if __name__ == '__main__':
    print("before replace")
    replace()
    print("after replace")
    



'''
if __name__ == '__main__':
    fname = '/public/software/apps/DeepLearning/Data/objects365/val/obj365_val_0000{:08d}.jpg' #sys.args[1] 
    #jname = open('/public/home/fengshen/private/CentripetalNet/mmdetection/Object365-x32/work_dirs/gloo_32x4_centripetalnet_mask_hg104/temp_0.json','r') #sys.args[2] 
    #jname = open('/public/home/fengshen/private/CentripetalNet/mmdetection/coco-eval/work_dirs/gloo_32x4_centripetalnet_mask_hg104/temp_0.json','r')
    jname = open('/public/home/fengshen/private/CentripetalNet/mmdetection/Object365-x32/work_dirs/gloo_32x4_centripetalnet_mask_hg104/temp_0.json','r')
    stats = json.load(jname)[0:20]
        
    last_id = -1
    for i in stats:
# 画矩形框
        img_id = i['image_id']
        if img_id != last_id:
            if last_id > 0:
                cv2.imwrite('{}.jpg'.format(last_id), img)
            last_id = img_id
            img = cv2.imread(fname.format(img_id))
        
        if(i['score'] < 0.2):
            continue

        x,y,w,h = list(map(int,i['bbox']))
        cls = i['category_id']
        print(x,y,x+w,y+h, "class is ", cls )
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 4)

# 标注文本
    #font = cv2.FONT_HERSHEY_SUPLEX
    #text = '001'
    #cv2.putText(img, text, (212, 310), font, 2, (0,0,255), 1)
    if last_id > 0:
        cv2.imwrite('{}.jpg'.format(last_id), img)
'''
