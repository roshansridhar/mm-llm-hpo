import requests
import base64
import json
from optimizers.base_optimizer import BaseOptimizer
import re


class GPT4VisionOptimizer(BaseOptimizer):
    def __init__(self, benchmarker, openai_api_key):
        super().__init__(benchmarker)
        self.api_key = openai_api_key

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def optimize(self, iterations, **kwargs):
        for iteration in range(iterations):
            plot_path = self.plot_validation_loss()
            plot_base64 = self.encode_image(plot_path) if plot_path else None
            prompt = self.generate_prompt(self.history)
            config = None
            while config is None:
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
                payload = {"model": "gpt-4-turbo",
                           "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                           "max_tokens": 100}
                if plot_base64:
                    payload['messages'][0]['content'].append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{plot_base64}"}})
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers,
                                         json=payload).json()
                try:
                    config = json.loads(response['choices'][0]['message']['content'])
                except json.decoder.JSONDecodeError:
                    try:
                        config = json.loads(
                            re.search(r'\{(.*?)\}', response['choices'][0]['message']['content'], re.DOTALL).group())
                    except:
                        continue  # If decoding fails, retry the request

            score = self.benchmarker.evaluate(config)
            self.history.append((iteration, config, score))
        return self.history
