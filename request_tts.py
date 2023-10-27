import requests
import numpy as np
import yaml


text = "你好，很高兴认识你"  # 要转换的文本

# 文本转语音，目前使用的是接口方式
def request_tts(text:str,aduio_path=None):
    with open('config.yaml','r') as file:
        cfg = yaml.safe_load(file)

    # 如果是接口方式使用
    url = cfg["api_url"]
    response = requests.post(url, json= {"text": text})
    if response.status_code == 200:
        data = response.json()
        if data["result"] == "Success":
            rate = data["rate"]
            audio = np.array(data["audio"], dtype=np.float32)
            if aduio_path != None:
                import scipy.io.wavfile as wavf
                wavf.write(aduio_path,rate, audio)
                print("tts转换完成：{}".format(aduio_path))
                return aduio_path
            else:
                print("tts转换完成")
                return audio 
        print("tts转换失败：{}".format(data["message"]))
        return None
    else:
        print("请求错误,status_code:{}", response.status_code)
        return None

if __name__ == '__main__':
    request_tts(text)
    