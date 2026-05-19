import requests
import pandas as pd
import time
import csv

chunk_size = 100
max_retries = 3
retry_delay = 2


model = "gpt-oss:120b"

def api_call(prompt):
    """
    Call the Onyxia LLM chat completion endpoint.

    Parameters
    ----------
    prompt : str
        User prompt to send to the remote chat-completions API.

    Returns
    -------
    str
        The model reply content extracted from the JSON response.

    Raises
    ------
    requests.RequestException
        If the HTTP request fails.
    KeyError
        If the expected JSON fields are missing in the response payload.
    """
    url = "https://llm.lab.sspcloud.fr/api/chat/completions"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": model,
      "messages": [
        {
          "role": "user",
          "content": prompt
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    # Sicherstellen, dass wir die Antwort im JSON-Format extrahieren
    response_json = response.json()
    # Die Antwort extrahieren
    reply = response_json["choices"][0]["message"]["content"]
    return reply

def get_chunk(df, chunk_number, chunk_size=chunk_size):
    start = chunk_number * chunk_size
    end = start + chunk_size
    return df.iloc[start:end]

def call_api_with_retry(prompt):
    for attempt in range(max_retries):
        try:
            return api_call(prompt)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt+1} nach Fehler: {e}")
                time.sleep(retry_delay)
            else:
                raise e

def process_chunk(df_chunk):
    results = []

    for index, row in df_chunk.iterrows():
        title = row["Titel"]
        #wz_code = row["SchlÃ¼ssel wz2025_urs"]
        wz_code = row.iloc[0]

        title_str = str(title) if pd.notna(title) else ""
        prompt_text = prompt.replace("<title>", title_str)

        try:
            generated_data = call_api_with_retry(prompt_text)  # Verwendet den Prompt mit dem Titel

            # Nach der Generierung sicherstellen, dass wir den Text zeilenweise aufteilen
            for data in generated_data.splitlines():
                results.append({
                    "Titel": title,
                    "wz_code": wz_code,
                    "Prompt": prompt_text,
                    "Generierte Daten": data,
                    "Error": None
                })

            print(f"Erfolgreich: {title}")

        except Exception as e:
            results.append({
                "Titel": title,
                "wz_code": wz_code,
                "Prompt": prompt_text,
                "Generierte Daten": None,
                "Error": str(e)
            })

            print(f"Fehler bei: {title} → {e}")

    return pd.DataFrame(results)