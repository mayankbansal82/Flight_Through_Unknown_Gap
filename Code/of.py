#!/usr/bin/env python

import getopt
import math
import numpy
import sys
import torch
# import torch2trt
import cv2
import numpy as np
import time

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'sintel-final' # 'sintel-final', or 'sintel-clean', or 'chairs-final', or 'chairs-clean', or 'kitti-final'
# arguments_strOne = './images/frame2.png'
# arguments_strTwo = './images/frame1.png'
# arguments_strOut = './out.flo'

# for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
#     if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use, see below
#     if strOption == '--one' and strArgument != '': arguments_strOne = strArgument # path to the first frame
#     if strOption == '--two' and strArgument != '': arguments_strTwo = strArgument # path to the second frame
#     if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super().__init__()
            # end

            def forward(self, tenInput):
                tenInput = tenInput.flip([1])
                tenInput = tenInput - torch.tensor(data=[0.485, 0.456, 0.406], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)
                tenInput = tenInput * torch.tensor(data=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

                return tenInput
            # end
        # end

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )
            # end

            def forward(self, tenInput):
                return self.netBasic(tenInput)
            # end
        # end

        self.netPreprocess = Preprocess()

        self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-spynet/network-' + arguments_strModel + '.pytorch', file_name='spynet-' + arguments_strModel).items() })

    # end

    def forward(self, tenOne, tenTwo):
        tenFlow = []

        tenOne = [ self.netPreprocess(tenOne) ]
        tenTwo = [ self.netPreprocess(tenTwo) ]

        for intLevel in range(5):
            if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
                tenOne.insert(0, torch.nn.functional.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2, count_include_pad=False))
                tenTwo.insert(0, torch.nn.functional.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2, count_include_pad=False))
            # end
        # end

        tenFlow = tenOne[0].new_zeros([ tenOne[0].shape[0], 2, int(math.floor(tenOne[0].shape[2] / 2.0)), int(math.floor(tenOne[0].shape[3] / 2.0)) ])

        for intLevel in range(len(tenOne)):
            tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            tenFlow = self.netBasic[intLevel](torch.cat([ tenOne[intLevel], OpticalFlow.backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
        # end

        return tenFlow
    # end
# end
##########################################################
# global backwarp_tenGrid
# backwarp_tenGrid = {}

class OpticalFlow():
    def __init__(self):
        
        self.netNetwork = None
        global backwarp_tenGrid
        backwarp_tenGrid = {}

    def backwarp(tenInput, tenFlow):
        if str(tenFlow.shape) not in backwarp_tenGrid:
            tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
            tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

            backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
        # end

        tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)), tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0)) ], 1)

        return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)
    # end

    ##########################################################

    def estimate(self, tenOne, tenTwo):
        if self.netNetwork is None:
            self.netNetwork = Network().cuda().eval()
        # end

        assert(tenOne.shape[1] == tenTwo.shape[1])
        assert(tenOne.shape[2] == tenTwo.shape[2])

        intWidth = tenOne.shape[2]
        intHeight = tenOne.shape[1]

        # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
        # assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

        tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
        tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        tenFlow = torch.nn.functional.interpolate(input=self.netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow[0, :, :, :].cpu()
    
    def find_gap_center(self,image_1,image_2):
        # i=0

        tenOne = torch.FloatTensor(numpy.ascontiguousarray(image_1[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(numpy.ascontiguousarray(image_2[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

        tenOutput = self.estimate(tenOne, tenTwo)

        # Calculate the magnitude and angle of the optical flow vectors
        tenOutput = tenOutput.cpu().numpy().transpose(1, 2, 0)
        magnitude, angle = cv2.cartToPolar(tenOutput[..., 0], tenOutput[..., 1])
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Convert angle to the hue value and set saturation to maximum
        hue = angle * 180 / np.pi / 2
        saturation = np.ones_like(magnitude) * 255

        # Create an HSV image
        hsv = np.zeros((tenOutput.shape[0], tenOutput.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = hue
        hsv[..., 1] = saturation
        hsv[..., 2] = magnitude

        # Convert HSV to RGB
        flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Save the optical flow visualization as an image
        # cv2.imshow('optical_flow_visualization', flow_color)
        
        # Calculate the magnitude of the optical flow vectors (assuming it's a 2-channel image)
        # magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Normalize the magnitude to range 0 to 255 for Otsu's thresholding
        # This step is important because Otsu's method is typically used on grayscale images with values from 0 to 255
        norm_magnitude = magnitude.astype(np.uint8)

        cv2.imwrite("./frames/norm_mag_" + str(int(time.time())) + ".png",norm_magnitude)
        # i+=1

        # Apply Otsu's thresholding method to determine the best threshold automatically
        # The function returns the threshold value and the thresholded image
        _, binary_mask = cv2.threshold(norm_magnitude, 5, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # Invert the mask to get black areas as white
        inverted_mask = cv2.bitwise_not(binary_mask)

        # edges = cv2.Canny(norm_magnitude,10,20)


        # Find all contours in the binary mask
        contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_gap = True
        # Check if any contours were found
        if contours:
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)

            (x,y),radius = cv2.minEnclosingCircle(largest_contour)
            if radius > 450:    
                found_gap = False

            
            # Create an empty mask for the largest contour
            # largest_contour_mask = np.zeros_like(binary_mask)
            largest_contour_mask = np.zeros_like(norm_magnitude)
            
            # Draw the largest contour on the mask with white color and filled
            cv2.drawContours(largest_contour_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
            
            # Save the mask of the largest contour
            largest_contour_mask_path = 'gap.png'  # Replace with your desired path
            # cv2.imwrite(largest_contour_mask_path, largest_contour_mask)
            

            # Compute the moments of the largest contour
            M = cv2.moments(largest_contour_mask)

            # Calculate the center (centroid) of the contour
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # This can happen if the contour is a single point or line
                cx, cy = 0, 0

            mag_bgr = cv2.cvtColor(norm_magnitude, cv2.COLOR_GRAY2BGR)
            contour_image_bgr = cv2.cvtColor(largest_contour_mask, cv2.COLOR_GRAY2BGR)
            invert_bgr = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)
            cv2.circle(contour_image_bgr, (cx,cy), 5, (0, 0, 255), -1)  # Red color

            # cv2.imshow('gap', largest_contour_mask
        else:
            found_gap = False

        return found_gap,(cx,cy), invert_bgr, mag_bgr, contour_image_bgr

        # image_1 = image_2
        # for i in range(5):
        #     success,image_2 = vidcap.read()

    # end

##########################################################

# if __name__ == '__main__':
#     of_ins = OpticalFlow()
#     vidcap = cv2.VideoCapture('./images/drone1.mp4')
#     success,image_1 = vidcap.read()
#     success,image_2 = vidcap.read()

#     while success:
#         tenOne = torch.FloatTensor(numpy.ascontiguousarray(image_1[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
#         tenTwo = torch.FloatTensor(numpy.ascontiguousarray(image_2[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

#         tenOutput = of_ins.estimate(tenOne, tenTwo)

#         # Calculate the magnitude and angle of the optical flow vectors
#         tenOutput = tenOutput.cpu().numpy().transpose(1, 2, 0)
#         magnitude, angle = cv2.cartToPolar(tenOutput[..., 0], tenOutput[..., 1])
#         magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

#         # Convert angle to the hue value and set saturation to maximum
#         hue = angle * 180 / np.pi / 2
#         saturation = np.ones_like(magnitude) * 255

#         # Create an HSV image
#         hsv = np.zeros((tenOutput.shape[0], tenOutput.shape[1], 3), dtype=np.uint8)
#         hsv[..., 0] = hue
#         hsv[..., 1] = saturation
#         hsv[..., 2] = magnitude

#         # Convert HSV to RGB
#         flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#         # Save the optical flow visualization as an image
#         cv2.imshow('optical_flow_visualization', flow_color)
        
#         # Calculate the magnitude of the optical flow vectors (assuming it's a 2-channel image)
#         # magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

#         # Normalize the magnitude to range 0 to 255 for Otsu's thresholding
#         # This step is important because Otsu's method is typically used on grayscale images with values from 0 to 255
#         norm_magnitude = magnitude.astype(np.uint8)

#         # Apply Otsu's thresholding method to determine the best threshold automatically
#         # The function returns the threshold value and the thresholded image
#         _, binary_mask = cv2.threshold(norm_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


#         # Invert the mask to get black areas as white
#         inverted_mask = cv2.bitwise_not(binary_mask)

#         # Find all contours in the binary mask
#         contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Check if any contours were found
#         if contours:
#             # Find the largest contour based on area
#             largest_contour = max(contours, key=cv2.contourArea)
            
#             # Create an empty mask for the largest contour
#             largest_contour_mask = np.zeros_like(binary_mask)
            
#             # Draw the largest contour on the mask with white color and filled
#             cv2.drawContours(largest_contour_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
            
#             # Save the mask of the largest contour
#             largest_contour_mask_path = 'gap.png'  # Replace with your desired path
#             # cv2.imwrite(largest_contour_mask_path, largest_contour_mask)
            

#             # Compute the moments of the largest contour
#             M = cv2.moments(largest_contour_mask)

#             # Calculate the center (centroid) of the contour
#             if M["m00"] != 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#             else:
#                 # This can happen if the contour is a single point or line
#                 cx, cy = 0, 0

            
#             cv2.circle(largest_contour_mask, (cx,cy), 5, (0, 0, 255), -1)  # Red color

#             cv2.imshow('gap', largest_contour_mask)
#         cv2.waitKey(0)

#         image_1 = image_2
#         for i in range(5):
#             success,image_2 = vidcap.read()

#         # objOutput = open(arguments_strOut, 'wb')

#         # numpy.array([ 80, 73, 69, 72 ], numpy.uint8)
#         # numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32).tofile(objOutput)
#         # numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)

#         # cv2.DenseOpticalFlow()

#         # objOutput.close()
# # end