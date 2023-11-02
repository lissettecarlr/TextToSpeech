import requests
from dotenv import load_dotenv
import os
from tqdm import tqdm

def ai_post(ip,token,prompt,model="Qwen-7B"):
    url = ip + "/v1/chat/completions"
    post_headers= {
            'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/json'
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
  
    try:
        response = requests.post(url,headers=post_headers,json=payload)
    except Exception as e:
        print("请求失败：{}".format(e))
        return None
    
    if response.status_code != 200:
        # 打印报错信息
        print("响应错误：{}   {}".format(response.status_code,response.text))
        return None 
    
    response_data = response.json()["choices"][0]["message"]["content"]
    return response_data

def fix_text(text_path):
    api_token = os.getenv("api_token")
    url = os.getenv("api_url_base")

    with open(text_path,'r',encoding='utf-8') as file:
        lines=  file.readlines()
    
    skip_lines = 0
    if os.path.exists('fix.txt'):
        with open('fix.txt','r',encoding='utf-8') as file:
            skip_lines = len(file.readlines())

    with open('fix.txt', 'a',encoding='utf-8') as file:
        for line in tqdm(lines[skip_lines:]):  
        #for line in lines[skip_lines:]:
            prompt = """
            输入是一段文本，你需要将其中的繁体字改成简体字，为句子添加标点符号。
            不要删除输入句子中的文本，将进过修改的文本完整输出，示例如下:
            示例输入：./custom_character_voice/paimon/processed_10568.wav|paimon|[ZH]喂 寫故事的人 你現在在附近嗎 你能幫行秋想想辦法嗎[ZH]
            示例输出：./custom_character_voice/paimon/processed_10568.wav|paimon|[ZH]喂！写故事的人，你现在在附近吗？你能帮行秋想想办法吗？[ZH]
            下面是真实的输入文本：
            """
            line = line.strip()
            prompt = prompt + line
            #print(prompt)
            res = ai_post(url,api_token,prompt)
            res = res.strip()
            if res == None:
                print("AI请求失败：{}".format(line))
                file.write(line+"\n")
            else:
                print(line)
                print(res)
                file.write(res + "\n")
    
load_dotenv()
fix_text("../short_character_anno.txt")
