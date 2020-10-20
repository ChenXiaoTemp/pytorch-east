import model.model as md
import cv2 as cv
import torch
from visualization.gradcam import GradCam
from visualization.misc_functions import save_class_activation_images

if __name__=='__main__':
    model=md.east_reset18()

    for name,param in model.named_parameters():
        print("Name {}: param: {}",name,param)

    img = cv.imread("/Users/shawn/Develop/WorkSpace/Files/img_11.jpg")
    input = torch.from_numpy(img).permute(2, 0, 1).float()
    print(input.size())
    input = torch.stack([input])
    output, loss = model(input)






    print(output)
