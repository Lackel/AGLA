import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import os
# os.environ['http_proxy'] = 'http://202.117.43.244:10007'
# os.environ['https_proxy'] = 'http://202.117.43.244:10007'

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, '..')
from llava.utils import disable_torch_init
from PIL import Image
import math
from lavis.models import load_model_and_preprocess
from sample import evolve_agla_sampling
from torchvision import transforms
from lavis.common.registry import registry
from augmentation import augmentation 
evolve_agla_sampling()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # model_class = registry.get_model_class('blip_image_text_matching')
    # model_class.PRETRAINED_MODEL_CONFIG_DICT['large'] = '/workspace/model/blip_itm_large/blip_itm_large.yaml'
    model_itm, image_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
    loader = transforms.Compose([transforms.ToTensor()])


    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]
        # For POPE
        prompt = question +  " Please answer this question with one word."
        # For generative tasks and MME
        # conv.append_message(conv.roles[0],  qs)
        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


        if args.use_agla:
            tensor_image = loader(raw_image.resize((384,384)))
            image = image_processors["eval"](raw_image).unsqueeze(0).to(device)
            question = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(question, padding='longest', truncation=True, return_tensors="pt").to('cuda')
            augmented_image = augmentation(image, question, tensor_image, model_itm, tokenized_text, raw_image)
            image_tensor_cd = vis_processors["eval"](augmented_image).unsqueeze(0).to(device)
        else:
            image_tensor_cd = None      

        with torch.inference_mode():
            outputs = model.generate({"image": image_tensor, "prompt": prompt},
                use_nucleus_sampling=True, num_beams=1,
                top_p = args.top_p, repetition_penalty=1,
                images_cd=image_tensor_cd, cd_alpha = args.alpha, cd_beta = args.beta, temperature=args.temperature)


        outputs = outputs[0]
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "model_id": "instruct_blip",
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-folder", type=str, default="/workspace/data/val2014")
    parser.add_argument("--question-file", type=str, default="/workspace/data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answers-file", type=str, default="/workspace/eval/output/test.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--use_agla", action='store_true', default=True)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
