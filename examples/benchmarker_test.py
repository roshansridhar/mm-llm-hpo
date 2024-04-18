from benchmarkers.benchmarker import Benchmarker
import json

task_id_path = "task_ids.json"

with open(task_id_path, "r") as file:
    task_dict = json.load(file)


for dataset, id in task_dict.items():
    for model in ["svm", "xgb"]:
        bm_obj = Benchmarker(id, model)
        search_space = bm_obj.get_search_space()
        print(dataset)
        print(bm_obj.task_id)
        print(bm_obj.model_name)
        print(search_space)
        sample_config = bm_obj.config_space.sample_configuration()
        print(sample_config)
        loss_log = bm_obj.evaluate(sample_config)
        print(loss_log)
    break