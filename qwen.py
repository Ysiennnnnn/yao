from http import HTTPStatus
import dashscope

dashscope.api_key='sk-64de2eee2c81419587d35b5fb6b5394f'
messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '如何做西红柿鸡蛋？'}]

response = dashscope.Generation.call(
    model='qwen-turbo',
    messages=messages,
    result_format='message',  # set the result to be "message" format.
)

if response.status_code == HTTPStatus.OK:
    print(response)
else:
    print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
        response.request_id, response.status_code,
        response.code, response.message
    ))