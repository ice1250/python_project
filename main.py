from dotenv import load_dotenv
import os
import openai

# load .env file
load_dotenv()

open_ai_key = os.getenv("OPEN_AI_KEY")

client = openai.OpenAI(api_key=open_ai_key)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "당신은 헤이마트의 상담원 입니다."},
        {"role": "user", "content": "안녕하세요"},
    ],
)

print(completion.choices[0].message)
