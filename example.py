from kuontts import TTS

if __name__ == "__main__":
    tts = TTS()
    res = tts.convert(text="你好，很高兴认识你",save_path="test.wav")
    print(res)
