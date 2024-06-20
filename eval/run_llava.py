import argparse
import json
from tqdm import tqdm
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria    
from PIL import Image
from transformers import set_seed
from sample import evolve_agla_sampling
evolve_agla_sampling()
from augmentation import augmentation
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from lavis.common.registry import registry 


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    device = 'cuda'
    model_class = registry.get_model_class('blip_image_text_matching')
    model_class.PRETRAINED_MODEL_CONFIG_DICT['large'] = '/workspace/model/blip_itm_large/blip_itm_large.yaml'
    model_itm, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
    loader = transforms.Compose([transforms.ToTensor()])
    
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]
        cur_prompt = question
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n'  + question

        conv = conv_templates[args.conv_mode].copy()
        # For POPE
        conv.append_message(conv.roles[0],  qs + " Please answer this question with one word.")
        # For generative tasks and MME
        # conv.append_message(conv.roles[0],  qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')

        raw_image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_agla:
            tensor_image = loader(raw_image.resize((384,384)))
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            question = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(question, padding='longest', truncation=True, return_tensors="pt").to('cuda')
            augmented_image = augmentation(image, question, tensor_image, model_itm, tokenized_text, raw_image)
            image_tensor = image_processor.preprocess(augmented_image, return_tensors='pt')['pixel_values'][0]
        else:
            image_tensor = None
        

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=raw_image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor.unsqueeze(0).half().cuda() if image_tensor is not None else None),
                cd_alpha = args.alpha,
                cd_beta = args.beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True)
            
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workspace/model/llava_1.5_7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/workspace/data/val2014")
    parser.add_argument("--question-file", type=str, default="/workspace/10_AGLA/data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answers-file", type=str, default="/workspace/10_AGLA/eval/output/test.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--use_agla", action='store_true', default=True)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
