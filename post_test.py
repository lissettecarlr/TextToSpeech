import requests
import numpy as np

url = "http://127.0.0.1:20004/tts/convert"  # 接口URL
text = "你好，很高兴认识你"  # 要转换的文本

payload = {"text": text}  # 构造请求参数
response = requests.post(url, json=payload)  # 发送POST请求
if response.status_code == 200:
    data = response.json()
    if data["result"] == "Success":
        rate = data["rate"]
        audio = np.array(data["audio"], dtype=np.float32)
        import scipy.io.wavfile as wavf
        wavf.write("123.wav",rate, audio)
        print("转换完成：{}".format("123.wav"))
    else:
        print("转换失败：{}".format(data["message"]))
else:
    print("请求错误:", response.status_code)
