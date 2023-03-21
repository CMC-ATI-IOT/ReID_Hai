## Training
Chọn file config tương ứng với backbone mà bạn muốn dùng tại đường dẫn /configs/Market1501/

Ví dụ với backbonw regvgg training trên 8 bộ data:
```
/configs/Market1501/repvgg_8data.yml
```

Thay đổi pretrain path tại file /home/tuantran/AI_TEAM/REID_HAI/fastreid/config/defaults.py, line 58
Ví dụ:
```
_C.MODEL.BACKBONE.PRETRAIN_PATH = '/home/tuantran/.cache/torch/hub/checkpoints/RepVGG-B3g4-200epochs-train.pth'
```

