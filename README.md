## Training
Chọn file config tương ứng với backbone mà bạn muốn dùng tại đường dẫn /configs/Market1501/

Ví dụ với backbone regvgg training trên 8 bộ data:
```
/configs/Market1501/repvgg_8data.yml
```
Tại đây, ta cũng có thể điều chỉnh path cho folder log (log ra cấu trúc mạng, các tham số và weights)


Với mỗi backbone, cần thay đổi các tham số phần BACKBONE tại file 
```
/home/tuantran/AI_TEAM/REID_HAI/configs/Base-bagtricks.yml
```
sao cho phù hợp với backbone đó.


Thay đổi pretrain path tại file /home/tuantran/AI_TEAM/REID_HAI/fastreid/config/defaults.py, line 58

Ví dụ:
```
_C.MODEL.BACKBONE.PRETRAIN_PATH = '/home/tuantran/.cache/torch/hub/checkpoints/RepVGG-B3g4-200epochs-train.pth'
```

Training trên 8 bộ dataset, backbone RepVGG B3g4:
```
python3 train.py --config-file ./configs/Market1501/repvgg.yml MODEL.DEVICE "cuda:0"
```

## Wandb
Tại file train.py, thay đổi các thông tin model mà bạn muốn log ra:
```
# setting config to log on wandb
if args.eval_only:
    configs = {
        "batch_size": cfg.TEST.IMS_PER_BATCH,

    }
else:
    configs = {
        "epochs": cfg.SOLVER.MAX_EPOCH,
        "learning_rate_init": cfg.SOLVER.BASE_LR,
        "batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "backbone": "RepVGG_B3g4",
    }

wandb.init(project = "ReID_Hai", entity = "ai-iot", config=configs, name = "RepVGG_B3g4_8data 1")
```

