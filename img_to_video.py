import os
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    parent_dir='save/exp2'
    sub_dir=['style-liif_v7']
    vid_name='styleliif_v1_Rreg_330-507kiter.mp4'
    img_list_all=[]
    for sub in sub_dir:
        full_dir=os.path.join(parent_dir,sub)
        img_list_all+=[os.path.join(full_dir,i) for i in os.listdir(full_dir)
                       if (i.endswith('.png') and i.split('.')[-2].endswith('000'))]

    img_list_all=list(set(img_list_all))
    img_list_all.sort()
    print(img_list_all)

    # img_list_all=img_list_all[:200]
    video=cv2.VideoWriter(os.path.join(parent_dir,vid_name),cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),25,(2066,2066))
    for i in tqdm(img_list_all):
        img=cv2.imread(i)
        video.write(img)

    video.release()
