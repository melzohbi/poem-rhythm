from prompts import num_tokens_from_string, prompt_text_250, prompt_text_500, prompt_text_750, prompt_text_1000
from openai import OpenAI

# check the number of tokens in the prompt text
# number_of_tokens = num_tokens_from_string(prompt_text_10, "gpt-4-0613")
# print(number_of_tokens)

GPT_MODEL = "gpt-4-0613"
client = OpenAI(api_key='..YOUR API KEY..')


def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


messages = []
messages.append(
    {"role": "system", "content": "You are an expert in language, music, phonetics and phonology."})

# batch the prompts to avoid token limit
# messages.append({"role": "user", "content": prompt_text_250})
# messages.append({"role": "user", "content": prompt_text_500})
# messages.append({"role": "user", "content": prompt_text_750})
messages.append({"role": "user", "content": prompt_text_1000})

chat_response = chat_completion_request(messages)
assistant_message = chat_response.choices[0].message

# save the assistant message to a text file
with open("gpt-4-results/assistant_message_1000.txt", "w") as file:
    file.write(assistant_message.content)
