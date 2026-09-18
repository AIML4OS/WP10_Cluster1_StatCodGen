
from statcodgen.synt_data_maker import SyntDataGenerator
import requests

class OnyxiaSyntDataGenerator(SyntDataGenerator):

    def api_call(self, prompt):
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
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
          "model": f"{self.model}",
          "messages": [
            {
              "role": "user",
              "content": prompt
            }
          ]
        }
        response = requests.post(url, headers=headers, json=data)
        reply = response.json()["choices"][0]["message"]["content"]
        return reply