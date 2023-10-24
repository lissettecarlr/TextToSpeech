# 文本转语音

本说明是推理部分，训练文档请见[这里](./finetune/README.md)

## 环境

* conda虚拟环境
    ```bash
    conda create -n vits python=3.10
    conda activate vits
    ```
* CUDA 11.8
* pytorch
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

* 进入kuontts文件夹安装软件包
    ```
    pip install -r ./kuontts/requirements.txt
    ```

    在windows上matplotlib会报错：
    ```
    ERROR: Could not build wheels for matplotlib, which is required to install pyproject.toml-based projects
    ```
    在requirements.txt中移除`matplotlib==3.3.1`,然后去[这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib)下载对于的包后手动安装

* 安装monotonic align
    ```base
    cd ./kuontts/monotonic_align
    mkdir monotonic_align
    python setup.py build_ext --inplace
    ```    
    如果是windows上则将build文件夹中的core.cp310-win_amd64.pyd复制到monotonic_align/monotonic_align中

## 导入模型

将模型文件放置到kuontts/OUTPUT_MODEL文件夹中，包含config.json和G_latest.pth两个文件

## 使用

```bash
python example.py
```
将会在当前目录生成名为test.wav的音频文件

api:
在config.yaml中设置端口，然后启动服务
```bash
python api.py
```

接口：
* 接口地址：http://127.0.0.1:20004/tts/convert
* 请求方法： POST
* 接口描述：问答
* 请求头
    * Content-Type: application/json
    * (暂时没有) Authorization: Bearer qmdr-xxxx
* 请求参数：
    * test (string): 需要转化的文本
    * language (string): 文本语言
    * speed (float): 语言速度

请求示例见文件[post_test.py](./post_test.py)