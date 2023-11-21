## 1 环境

### 通用环境
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
### 安装包
```bash
conda install -c conda-forge ffmpeg

pip install --upgrade --force-reinstall regex
pip install --force-reinstall soundfile
pip install --force-reinstall gradio
pip install imageio==2.4.1
pip install --upgrade youtube-dl
pip install moviepy

pip install --no-build-isolation -r requirements.txt
pip install --upgrade numpy
pip install --upgrade --force-reinstall numba
pip install --upgrade Cython

pip install --upgrade pyzmq
pip install pydantic==1.10.4
pip install ruamel.yaml

mkdir segmented_character_voice
```

**11月whisper发发布了'large-v3'模型，如果要使用新模型来转换语音**
```bash
将之前的卸载
pip uninstall whisper
再安装
pip install git+https://github.com/openai/whisper.git
```
验证:
```bash
>>> whisper.available_models()
['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large']
```
之后将下列的脚本模型选择改成`large-v3`即可。

#### 安装monotonic align
```base
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```

### 下载微调额外数据（大概三百M）
```bash
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/sampled_audio4ft_v2.zip
unzip sampled_audio4ft_v2.zip
```

## 2 模型训练

代码来源于[VITS-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning/blob/main/README_ZH.md)

### 下载预训练模型
模型分为三个文件D_0.pth、G_0.pth和finetune_speaker.json，前面两个放入到pretrained_models中，后面一个放到configs文件夹中。
执行下列命令会自动下载和存储
```bash
mkdir pretrained_models

# 中日双语模型
wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/D_0-p.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/G_0-p.pth -O ./pretrained_models/G_0.pth
wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/config.json -O ./configs/finetune_speaker.json

# 中日英
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/D_trilingual.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth -O ./pretrained_models/G_0.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/configs/uma_trilingual.json -O ./configs/finetune_speaker.json

# 纯中文
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/D_0.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/G_0.pth -O ./pretrained_models/G_0.pth
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/config.json -O ./configs/finetune_speaker.json
```

以下是使用纯中文模型


### 训练数据

### 短音频

* 2秒以上，10秒以内，尽量不要有背景噪音。
* 一个角色至少10条，最好每个角色20条以上
* 格式:
    ```
    Your-zip-file.zip
    ├───Character_name_1
    ├   ├───xxx.wav
    ├   ├───...
    ├   ├───yyy.mp3
    ├   └───zzz.wav
    ```
* zip存入`./custom_character_voice/`文件夹中

### 长音频

* 命名格式为：{CharacterName}_{random_number}.wav
* 必须是.wav文件
* 存入到`./raw_audio/`文件夹中

### 视频
* 命名格式为：{CharacterName}_{random_number}.mp4
* 必须是.mp4
* CharacterName必须是英文字
* 存入到`./video_data/`文件夹中


### 数据处理
对所有上传的数据进行自动去背景音和标注，需要调用Whisper和Demucs，根据情况选择执行

```bash
# 将所有视频抽取音频
python scripts/video2audio.py
# 将所有音频去噪，文件被存放在denoised_audio中
python scripts/denoise_audio.py
# 分割并标注长音频，从./denoised_audio/文件夹中加载音频，输出到segmented_character_voice中
python scripts/long_audio_transcribe.py --languages "C" --whisper_size large
# 标注短音频
python scripts/short_audio_transcribe.py --languages "C" --whisper_size large
# 底模采样率可能与辅助数据不同，需要重采样
python scripts/resample.py
```

这里会生成`xx_character_anno.txt`的标注文件，如果想提升之后训练质量，可以对内容进行校准，使其实际音频和whisper识别出来的一致。

下面将会根据`xx_character_anno.txt`转化最终标准，如果总样本少于100条/样本质量一般或较差/样本来自爬取的视频，可以使用辅助训练数据，就是上面那个额外数据。

```bash
python preprocess_v2.py --add_auxiliary_data True --languages "C"
```

如果总样本量很大/样本质量很高/希望加速训练/只有二次元角色则可以直接
```bash
python preprocess_v2.py --languages "C"
```
生成标注文件会被保存在当前目录


### 训练

如果需要可视化
```bash
export TENSORBOARD_BINARY=/home/server/anaconda3/envs/vits/bin/tensorboard
tensorboard --logdir="./OUTPUT_MODEL" --bind_all
```

如果是首次训练
```bash
# 指定GPU
export CUDA_VISIBLE_DEVICES="0"
python finetune_speaker_v2.py -m "./OUTPUT_MODEL" --max_epochs "200" --drop_speaker_embed True 
```

如果是继续训练
```bash
python finetune_speaker_v2.py -m "./OUTPUT_MODEL" --max_epochs "200" --drop_speaker_embed False --cont True
```

* 根据显卡修改`modified_finetune_speaker.json`中的`batch_size`，我这里设置为32，消耗了13G显存。

### 测试结果
```bash
cp ./configs/modified_finetune_speaker.json ./finetune_speaker.json
python VC_inference.py --model_dir ./OUTPUT_MODEL/G_latest.pth --share True
```

### 成品

#### paimon

* 训练数据来源于huggingface，总10664条。
* 使用`python scripts/short_audio_transcribe.py --languages "C" --whisper_size large`对语音进行标注。
* 通过qwen-7b对标注的文本进行繁体转简体
* 通过`python preprocess_v2.py --languages "C"`转化标注
* 进行训练，batch_size设置为64时，显存需要至少24G。
* 结果见[度盘](https://pan.baidu.com/s/1SL5aqT4LqQXu2OU0_Ph40w?pwd=kuon)

### 报错解决

#### 1
在windows上安装matplotlib时会报错：
```bash
ERROR: Could not build wheels for matplotlib, which is required to install pyproject.toml-based projects
```
在requirements.txt中移除`matplotlib==3.3.1`,然后去[这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib)下载对于的包后手动安装，例如根目录下我已经下载了的
```bash
pip install .\matplotlib-3.5.2-cp310-cp310-win_amd64.whl
```