# 文本转语音

本说明是推理部分，训练文档请见[这里](./finetune/README.md)

## 1 环境

### 1.1 如果离线使用

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

* 进入kuontts.offline文件夹安装软件包
    ```
    pip install -r requirements.txt
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

### 1.2 如果是在线使用

客户端部分只需要

* requests
* numpy
* yaml

服务端部分请还是根据上面的离线环境安装

## 2 使用

### 2.1 导入模型

离线使用或者部署服务需要下载模型，放置到kuontts/offline/OUTPUT_MODEL文件夹中，如果不放在治理则需要在使用`OfflineTTS`时传入地址，包含config.json和G_latest.pth两个文件。

### 2.2 离线使用

修改kuontts中的配置文件config.yaml
```yaml
channel : offline
```
使用示例参考example.py，或者直接运行，在同级目录生成示例音频

### 2.3 服务部署

端口直接在api.py中修改，然后
```bash
python api.py
```
还有在代码中如果需要选择GPU，则修改
```bash
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### 2.4 在线使用

也就是调用接口请求了上面部署的服务，修改kuontts中的配置文件config.yaml
```yaml
channel : offline
api_url: "填入接口地址"
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
    * speaker(string):模型中的角色



## 3 其他

offline在`kuontts/offline/text/mandarin.py`中设置了
```python
jieba.setLogLevel(logging.INFO)
```
以取消显示下列内容
```bash
Building prefix dict from the default dictionary ...
DEBUG:jieba:Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
DEBUG:jieba:Loading model from cache /tmp/jieba.cache
Loading model cost 0.586 seconds.
DEBUG:jieba:Loading model cost 0.586 seconds.
Prefix dict has been built successfully.
DEBUG:jieba:Prefix dict has been built successfully.
```