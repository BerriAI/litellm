from datetime import datetime
import json
import os

dir_created = False

def get_data_file():
    global dir_created
    current_date = datetime.now()
    dir = '/app/custom_log/continue_development_data'
    if not dir_created:
        os.makedirs(dir, exist_ok=True)
        dir_created = True
    return f'{dir}/continue_development_data_{current_date.strftime('%Y%m')}.jsonl'


def add_continue_plugin_event(data: str):
    with open(get_data_file(), 'a', encoding='utf-8') as file:
        file.write(f'{data}\n')
