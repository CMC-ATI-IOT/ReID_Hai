## Training
Chọn file config tương ứng với backbone mà bạn muốn dùng tại đường dẫn /configs/Market1501/

Ví dụ với backbone regvgg training trên 8 bộ data:
```
/configs/Market1501/repvgg_8data.yml
```
Tại đây, ta cũng có thể điều chỉnh path cho folder log (log ra cấu trúc mạng, các tham số và weights)


Với mỗi backbone, cần thay đổi các tham số phần BACKBONE tại file /home/tuantran/AI_TEAM/REID_HAI/configs/Base-bagtricks.yml sao cho phù hợp với backbone đó.


Thay đổi pretrain path tại file /home/tuantran/AI_TEAM/REID_HAI/fastreid/config/defaults.py, line 58

Ví dụ:
```
_C.MODEL.BACKBONE.PRETRAIN_PATH = '/home/tuantran/.cache/torch/hub/checkpoints/RepVGG-B3g4-200epochs-train.pth'
```



