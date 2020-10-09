import model.model as md
import cv2 as cv
import torch

if __name__=='__main__':
    model=md.east_reset18()
    img = cv.imread("/Users/shawn/Develop/WorkSpace/Files/person.jpg")
    input=torch.from_numpy(img).permute(2,0,1).float()
    print(input.size())
    input=torch.stack([input])
    output=model(input)
    print(output.size())
