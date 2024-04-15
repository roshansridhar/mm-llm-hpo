import json
from PIL import Image
from optimizers.base_optimizer import BaseOptimizer
from transformers import AutoProcessor, LlavaForConditionalGeneration, pipeline


class LlavaOptimizer(BaseOptimizer):
    def __init__(self, benchmarker, model_id, quantization_config, device="cuda"):
        super().__init__(benchmarker)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map={0: device})
        self.device = device

    def optimize(self, iterations, image_path=None, **kwargs):
        for _ in range(iterations):
            prompt = self.generate_prompt(self.history)
            inputs = self.processor([prompt], images=Image.open(image_path), padding=True, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=200)
            generated_text = self.processor.batch_decode(output, skip_special_tokens=True)
            config = json.loads(generated_text[0].split("Config:")[-1].strip())
            score = self.benchmarker.evaluate(config)
            self.history.append((config, score))
        return self.history
