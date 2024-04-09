import numpy as np
import random
from scipy.special import comb
import torch
import torch.fft
import cv2
import matplotlib.pyplot as plt

class FIESTA(object):
    def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=100000):
        self.nPoints=nPoints
        self.nTimes=nTimes
        self.vrange=vrange
        self.background_threshold=background_threshold
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array([bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self,points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point,end_point=inputs.min(),inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints-2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random()<=inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        location = np.clip(location, self.vrange[0] - np.percentile(inputs, slide_limit), self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs*scale + location, self.vrange[0], self.vrange[1])  # 최종적으로는 0~1로 normalization 해줌

    def Local_Location_Scale_Augmentation(self, image, mask):
        output_image = np.zeros_like(image)

        mask = mask.astype(np.int32)

        output_image[mask == 0] = self.location_scale_transformation(self.non_linear_transformation(image[mask==0], inverse=True, inverse_prop=1))

        for c in range(1,np.max(mask)+1):
            if (mask==c).sum()==0:continue
            output_image[mask == c] = self.location_scale_transformation(self.non_linear_transformation(image[mask == c], inverse=True, inverse_prop=0.5))

        if self.background_threshold>=self.vrange[0]:
            output_image[image <= self.background_threshold] = image[image <= self.background_threshold]

        return output_image

    def phase_attention_with_Bilateral(self, amplitude, phase):
        constant_amplitude = torch.ones_like(amplitude)
        reconstructed_from_phase = torch.fft.ifftn(torch.polar(constant_amplitude, phase)).real
        reconstructed_from_phase = torch.sqrt(reconstructed_from_phase ** 2).detach().cpu().numpy()
        bilateral = cv2.bilateralFilter(src=reconstructed_from_phase, d=3, sigmaColor=75, sigmaSpace=75)
        phase_attention = (bilateral - np.min(bilateral)) / (np.max(bilateral) - np.min(bilateral))
        return np.expand_dims(phase_attention, axis=-1)

    def sector_mask(self, amplitude, center, radius, angle_range=(0, 360)):
        y, x = np.ogrid[:amplitude.shape[0], :amplitude.shape[1]]

        # Calculate the distance from the center
        distance_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        # Calculate the angle (in degrees) for each point with respect to center
        angles = np.rad2deg(np.arctan2(y - center[1], x - center[0])) % 360

        # Create mask where distance is less than the radius and within the angle range
        mask = (distance_from_center <= radius) & (angles >= angle_range[0]) & (angles <= angle_range[1])

        return mask

    def amplitude_masking(self, amplitude, phase):
        re_amp = amplitude.clone()

        angles = np.linspace(0, 2*np.pi, 100)
        h_theta = np.zeros_like(angles)

        for i, theta in enumerate(angles): # radian을 도로 변환  => rad * 180 / np.pi = 도
            sum_freq = 0
            for r in range(1, amplitude.shape[0]):
                x = int(r * np.cos(theta))
                y = int(r * np.sin(theta))
                sum_freq += amplitude[x, y]
            h_theta[i] = sum_freq

        h_theta = np.abs(h_theta)  # h_theta /= np.sum(h_theta)

        max_angle = angles[np.where(h_theta == h_theta.max())][0] * 180 / np.pi
        min_angle = angles[np.where(h_theta == h_theta.min())][0] * 180 / np.pi

        degree = np.random.randint(60)  # 최대 각도 60도로 설정
        center = (amplitude.shape[0] // 2, amplitude.shape[1] // 2)
        max_angle_start, max_angle_end = max_angle - (degree/2), max_angle + (degree/2)
        min_angle_start, min_angle_end = min_angle - (degree/2), min_angle + (degree/2)

        if max_angle_end > 360:
            max_angle_end = (max_angle_end - 360)
        max_angle_range = (max_angle_start, max_angle_end)

        if min_angle_end > 360:
            min_angle_end = (min_angle_end - 360)
        min_angle_range = (min_angle_start, min_angle_end)

        radius = np.random.randint(amplitude.shape[0] // 4)  # 최대 반지름 48로 설정

        ###### Min & Max masking code ######
        max_mask = 1 - self.sector_mask(amplitude, center, radius, max_angle_range)
        min_mask = 1 - self.sector_mask(amplitude, center, radius, min_angle_range)

        masked_amp = amplitude * torch.from_numpy(np.expand_dims(max_mask, axis=-1))
        masked_amp = torch.fft.ifftshift(masked_amp)
        max_mask_reconstruction = torch.fft.ifftn(torch.polar(masked_amp, phase)).real

        ##### Min & Max switching code #####
        max_coord, min_coord = np.where(max_mask == False), np.where(min_mask == False)
        if max_coord[0].shape < min_coord[0].shape:
            x_min_coord, y_min_coord = min_coord[0][:len(max_coord[0])], min_coord[1][:len(max_coord[1])]
            re_amp[x_min_coord, y_min_coord] = amplitude[max_coord]
            re_amp[max_coord] = amplitude[x_min_coord, y_min_coord]
        else:
            x_max_coord, y_max_coord = max_coord[0][:len(min_coord[0])], max_coord[1][:len(min_coord[1])]
            re_amp[x_max_coord, y_max_coord] = amplitude[min_coord]
            re_amp[min_coord] = amplitude[x_max_coord, y_max_coord]
        re_amp = torch.fft.ifftshift(re_amp)
        max_min_switching_reconstruction = torch.fft.ifftn(torch.polar(re_amp, phase)).real

        return max_mask_reconstruction, max_min_switching_reconstruction

    def FourierAugmentativeTransformer(self, image):  # image=[192, 192, 1]
        fft_res = torch.fft.fftn(torch.from_numpy(image), dim=(0, 1))
        amplitude = torch.abs(torch.fft.fftshift(fft_res))  # 0 주파수 성분을 이미지의 중앙으로 이동
        phase = torch.angle(fft_res)

        if 0.5 > np.random.random():  # Reversing the histogram distribution of the amplitude
            amplitude = (2 * np.median(amplitude)) - amplitude
        amp_masking, amp_intra_modulation = self.amplitude_masking(amplitude=amplitude, phase=phase)
        phase_attention = self.phase_attention_with_Bilateral(amplitude, phase)

        aug_img = (0.5 * amp_masking) + (0.5 * amp_intra_modulation)
        aug_img = aug_img * phase_attention + aug_img
        aug_img = aug_img.detach().cpu().numpy()

        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        aug_img = np.clip(aug_img * scale + location, self.vrange[0], self.vrange[1])
        aug_img = (image + aug_img) / 2
        return aug_img