import requests
import base64
import json
from optimizers.base_optimizer import BaseOptimizer
import re
from loguru import logger

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
            prompt = prompt.strip("Config:") + " The graph of validation loss versus hyperparameter values over previous trials is attached for your reference. Config:"
            config = None
            while config is None:
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
                payload = {"model": "gpt-4-turbo",
                           "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                           "temperature": 0,
                           "max_tokens": 2000,
                           "top_p": 1}
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
                        for parameter_name, parameter_value in config.items():
                            if parameter_value < self.benchmarker.search_space[parameter_name][0] or \
                                    parameter_value > self.benchmarker.search_space[parameter_name][1]:
                                logger.debug(f"config is outside of the search space. Request again. Config: {config}")
                                config = None
                                break
                    except:
                        logger.debug(f"response is not json format. Request again. Response: {response['choices'][0]['message']['content']}")
                        continue  # If decoding fails, retry the request
            print(prompt)
            print(response)
            score = self.benchmarker.evaluate(config)
            self.history.append((iteration, config, score))
            # self.history.append((iteration, {'C': 0.0009765625, 'gamma': 0.0009765625},
            #                      {'train_loss': 1.0, 'validation_loss': 0.29276500280426243,
            #                       'cost': 0.3861689567565918}))
        return self.history
