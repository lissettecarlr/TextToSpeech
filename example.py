from kuontts import TTS
import os

if __name__ == "__main__":
    tts = TTS()
    ofile = "test.wav"
    res = tts.convert(text="你好，很高兴认识你",save_path=ofile)
    if os.path.exists(ofile):
        print("已生成转化音频")
    else:
        print("失败")

    res = tts.convert(text="你好，很高兴认识你")
    print("rate:{}".format(res[0]))
    print("audio:{}".format(res[1]))