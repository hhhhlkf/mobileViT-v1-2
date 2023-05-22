## mobileViT-v1&2

简介：MobileViTv1&amp;2的简单复现 ——by pytorch

1. 在`train.py`脚本中将`--data-path`设置成解压后的`flower_photos`文件夹绝对路径
2. 下载预训练权重，在`model.py`文件中每个模型都有提供预训练权重的下载地址，根据自己使用的模型下载对应预训练权重
3. 在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径
4. 设置好数据集的路径`--data-path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)
5. 在`predict.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)
6. 在`predict.py`脚本中将`img_path`设置成你自己需要预测的图片绝对路径
7. 设置好权重路径`model_weight_path`和预测的图片路径`img_path`就能使用`predict.py`脚本进行预测了
8. 如果要使用自己的数据集，请按照花分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的`num_classes`设置成你自己数据的类别数
