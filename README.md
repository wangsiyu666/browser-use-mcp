# 需要启动chrome并监听端口 "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
# 这里也要设置以下，防止browser API 报错
os.environ["OPENAI_API_KEY"] = ""
# server端，client端 LLM API_KEY
