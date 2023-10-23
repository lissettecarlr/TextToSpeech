import numpy as np
import torch
from torch import no_grad, LongTensor
import os

import commons
from models import SynthesizerTrn
from text import text_to_sequence
import utils


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class TTS():
    def __init__(self,config_path = "./finetune_speaker.json",model_path = "./OUTPUT_MODEL/G_latest.pth") -> None:
        self.config_path = config_path
        self.model_path = model_path
        
        hps = utils.get_hparams_from_file(config_path)
        net_g = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).to(device)
        _ = net_g.eval()
        _ = utils.load_checkpoint(model_path, net_g, None)
        speaker_ids = hps.speakers
        self.tts_fn = TTS.create_tts_fn(net_g, hps, speaker_ids)
   
    @staticmethod
    def get_text(text, hps, is_symbol):
        text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    @staticmethod
    def create_tts_fn(model, hps, speaker_ids):
        def tts_fn(text, speaker, language, speed=1):
            '''
            language :  in  ['日本語', '简体中文', 'English']
            speed : minimum=0.1, maximum=5, default 1
            '''
            
            if language == "简体中文":
                text = "[ZH]" + text + "[ZH]"
            elif language == "日本語":
                text = "[JA]" + text + "[JA]"
            elif language == "English":
                text = "[EN]" + text + "[EN]"
            else:
                raise ValueError("language error")
             
            speaker_id = speaker_ids[speaker]
            stn_tst = TTS.get_text(text, hps, False)
            with no_grad():
                x_tst = stn_tst.unsqueeze(0).to(device)
                x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
                sid = LongTensor([speaker_id]).to(device)
                audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                    length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
            del stn_tst, x_tst, x_tst_lengths, sid

            return "Success", (hps.data.sampling_rate, audio)

        return tts_fn

    def run(self,text,speaker="kt",language='简体中文', speed=1,save_path:str=None):
        text_output,audio_output = self.tts_fn(text=text, speaker=speaker, language=language, speed=speed)
        if(save_path != None):
            if not save_path.endswith('.wav'):
                save_path = os.path.splitext(save_path)[0] + '.wav' 
            import soundfile as sf
            sf.write(save_path, audio_output[1], audio_output[0])
        return text_output,audio_output


if __name__ == "__main__":
    test = TTS()
    test.run(text="你好，很高兴认识你",save_path="test")
    # config_path = "./finetune_speaker.json"
    # model_path = "./OUTPUT_MODEL/G_latest.pth"
    # hps = utils.get_hparams_from_file(config_path)
    # net_g = SynthesizerTrn(
    #     len(hps.symbols),
    #     hps.data.filter_length // 2 + 1,
    #     hps.train.segment_size // hps.data.hop_length,
    #     n_speakers=hps.data.n_speakers,
    #     **hps.model).to(device)
    # _ = net_g.eval()
    # _ = utils.load_checkpoint(model_path, net_g, None)
    # speaker_ids = hps.speakers
    # tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    # speakers = list(hps.speakers.keys())

    # text_output,audio_output = tts_fn(text="你好，很高兴认识你", speaker="kt", language='简体中文', speed=1)
    # # 保存音频文件
    # import soundfile as sf
    # sf.write('test.wav', audio_output[1], audio_output[0])
