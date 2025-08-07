import csv
import os
from datetime import datetime

ANALYTICS_FILE = "chatbot_usage.csv"

def log_interaction(question, answer, language_code, response_time):
    file_exists = os.path.isfile(ANALYTICS_FILE)
    
    with open(ANALYTICS_FILE, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'question', 'answer', 'language', 'response_time_sec']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'language': language_code,
            'response_time_sec': round(response_time, 2)
        })
