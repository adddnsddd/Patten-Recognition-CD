# 人脸识别系统

[独立程序](../人脸识别系统.exe)：双击运行，无需安装 Python。

[最终课程报告](../文档/课程报告/人脸识别系统课程设计报告_统一格式.docx)

源码运行：使用本目录 .venv 中的 Python 执行 main.py，或双击启动程序.bat。换电脑时重新建立环境并安装 requirements.txt，不复制 .venv。程序启动不会自动开启摄像头；真实人脸指标尚待采集与测量。测试位于 tests，已有验证记录保留在 outputs 与 build。重新打包使用 build_portable.ps1。

公开数据实验：已下载 Extended Yale B 裁剪人脸子集并完成680张图片的 PCA/LBPH 实验，见 [实验结果](outputs/public_yale_b/实验结果.md) 和 [数据说明](datasets/extended_yale_b/README.md)。复现命令：`python run_public_dataset.py`。该实验与摄像头采集数据分开保存，不代表全部课程采集要求已满足。
