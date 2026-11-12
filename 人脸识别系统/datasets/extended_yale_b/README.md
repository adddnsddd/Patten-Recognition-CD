# 公开数据实验：Extended Yale B

原始数据说明：https://vision.ucsd.edu/datasets/extended-yale-face-database-b-b

实际下载备份：https://github.com/codehxj/CroppedYale （master 分支）

`selected_cropped.zip` 保存从备份选取的前 14 名人员的正面原始 PGM 文件，未人工生成、调暗或扩增。运行脚本另选 680 张开展实验；清单记录原路径、身份、集合和文件 SHA256。完整档案 SHA256 记录于 results.json。数据仅用于本地课程研究，提交或传播图片前应核对原始数据使用条款并引用原作者。

在人脸识别系统目录运行：`python run_public_dataset.py`。结果位于 `outputs/public_yale_b/`，独立于摄像头采集数据和现有模型。

固定随机种子 20261112；注册人员每人训练30、已知验证10、普通测试10、极端光照测试10；未知验证两人、未知测试另两人，每人20。极端光照按文件中的光源方位角与仰角绝对值之和排序，预留最大的10张；这表示极端照明条件，不等于人工标注的环境暗光。其余照片随机分组，普通测试仍含光照变化。

图像缩放至64×64并做直方图均衡，PCA固定20主成分，LBPH使用项目默认配置。阈值仅由验证数据选择，测试集不参与调参。身份以P/U编号匿名化，原数据身份可由清单追溯。

## 与课程采集要求的差异

这是图像互斥的公开数据识别实验，不能声称完全满足截图规定的真实采集实验：没有独立采集批次证据；没有侧脸、眼镜/遮挡分组，也没有20张无人背景图；已裁剪人脸不经过摄像头人脸检测流程。因此不写入正式采集清单，不将划分编号冒充采集批次，不据此宣称摄像头检测召回率或课程全部验收通过。

参考：Georghiades A S, Belhumeur P N, Kriegman D J. From Few to Many: Illumination Cone Models for Face Recognition under Variable Lighting and Pose. IEEE TPAMI, 2001, 23(6): 643–660. Lee K C, Ho J, Kriegman D J. Acquiring Linear Subspaces for Face Recognition under Variable Lighting. IEEE TPAMI, 2005, 27(5): 684–698.
