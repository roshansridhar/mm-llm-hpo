import requests
import base64
import json
from optimizers.base_optimizer import BaseOptimizer


class GPT4VisionOptimizer(BaseOptimizer):
    def __init__(self, benchmarker, openai_api_key):
        super().__init__(benchmarker)
        self.api_key = openai_api_key

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def optimize(self, iterations, image_path=None, **kwargs):
        base64_image = self.encode_image(image_path)
        image_info = f"\nImage as base64 for reference: {base64_image}\n"  # Customized additional info
        for _ in range(iterations):
            prompt = self.generate_prompt(self.history, additional_info=image_info)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 300
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
            config = json.loads(response['choices'][0]['message']['content'])
            score = self.benchmarker.evaluate(config)
            self.history.append((config, score))
        return self.history
