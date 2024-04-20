import json
from PIL import Image
from optimizers.base_optimizer import BaseOptimizer
from transformers import AutoProcessor, LlavaForConditionalGeneration


class LlavaOptimizer(BaseOptimizer):
    def __init__(self, benchmarker):
        super().__init__(benchmarker)
        model_id = "llava-hf/llava-1.5-7b-hf"
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16
        # )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto",
                                                                   # quantization_config=quantization_config
                                                                   )

    def optimize(self, iterations, **kwargs):
        for iteration in range(iterations):
            prompt = self.generate_prompt(self.history)
            plot_path = self.plot_validation_loss()
            plot_image = Image.open(plot_path) if plot_path else None
            config = None
            while config is None:
                inputs = self.processor([prompt], images=[plot_image], padding=True, return_tensors="pt")
                output = self.model.generate(**inputs, max_new_tokens=100)
                generated_text = self.processor.batch_decode(output, skip_special_tokens=True)
                try:
                    config = json.loads(generated_text[0].split("Config:")[-1].strip())
                except json.decoder.JSONDecodeError:
                    continue  # If decoding fails, retry generation

            score = self.benchmarker.evaluate(config)
            self.history.append((iteration, config, score))
        return self.history
