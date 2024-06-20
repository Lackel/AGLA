import torch
import numpy as np
from lavis.common.gradcam import getAttMap
from torchvision import transforms
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from torchvision import transforms

# Generating augmentated images based on the input prompt
def augmentation(image, question, tensor_image, model, tokenized_text, raw_image):
    with torch.set_grad_enabled(True):
        gradcams, _ = compute_gradcam(model=model,
                            visual_input=image,
                            text_input=question,
                            tokenized_text=tokenized_text,
                            block_num=6)
    gradcams = [gradcam_[1] for gradcam_ in gradcams]
    gradcams1 = torch.stack(gradcams).reshape(image.size(0), -1)
    itc_score = model({"image": image, "text_input": question}, match_head='itc')
    ratio = 1 - itc_score/2
    ratio = min(ratio, 1-10**(-5))
    resized_img = raw_image.resize((384, 384))
    norm_img = np.float32(resized_img) / 255
    gradcam = gradcams1.reshape(24,24)


    avg_gradcam = getAttMap(norm_img, gradcam.cpu().numpy(), blur=True, overlap=False)
    temp, _ = torch.sort(torch.tensor(avg_gradcam).reshape(-1), descending=True)
    cam1 = torch.tensor(avg_gradcam).unsqueeze(2)
    cam = torch.cat([cam1, cam1, cam1], dim=2)

    mask = torch.where(cam<temp[int(384 * 384 * ratio)], 0, 1)
    new_image = tensor_image.permute(1, 2, 0) * mask
    unloader = transforms.ToPILImage()
    imag = new_image.clone().permute(2, 0, 1) 
    imag = unloader(imag)
    
    return imag