from PIL import Image
from pathlib import Path
import os, glob, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_mse, compare_psnr, compare_ssim
import csv
from statistics import mean, median, variance, stdev
from natsort import natsorted


class ResultArrangement(object):
    def __init__(self, path_to_project, individual_img):
        self.path_to_project = './' + path_to_project

        self.target_for_arranging = ['real_A', 'fake_B', 'rec_A']
        self.target_for_deleting = ['real_B', 'fake_A', 'rec_B']

        self.individual_img = individual_img

    def arrange(self):
        self.mkdir(self.path_to_project, self.target_for_arranging)
        files = glob.glob(self.path_to_project + '/images/*.png')

        for file in files:
            img = cv2.imread(file)
            get_file_name = os.path.basename(file)
            print(get_file_name)

            if file.find(self.target_for_arranging[0]) != -1:
                output_name = get_file_name.rstrip('_{}.png'.format(self.target_for_arranging[0]))
                cv2.imwrite('{}/{}/{}.jpg'.format(self.path_to_project, self.target_for_arranging[0], output_name), img)

            if file.find(self.target_for_arranging[1]) != -1:
                output_name = get_file_name.rstrip('_{}.png'.format(self.target_for_arranging[1]))
                cv2.imwrite('{}/{}/{}.jpg'.format(self.path_to_project, self.target_for_arranging[1], output_name), img)

            if file.find(self.target_for_arranging[2]) != -1:
                output_name = get_file_name.rstrip('_{}.png'.format(self.target_for_arranging[2]))
                cv2.imwrite('{}/{}/{}.jpg'.format(self.path_to_project, self.target_for_arranging[2], output_name), img)

    @staticmethod
    def mkdir(path_to_project, target_for_arranging):
        if not os.path.exists(path_to_project + '/{}'.format(target_for_arranging[0])):
            os.mkdir(path_to_project + '/{}'.format(target_for_arranging[0]))
        if not os.path.exists(path_to_project + '/{}'.format(target_for_arranging[1])):
            os.mkdir(path_to_project + '/{}'.format(target_for_arranging[1]))
        if not os.path.exists(path_to_project + '/{}'.format(target_for_arranging[2])):
            os.mkdir(path_to_project + '/{}'.format(target_for_arranging[2]))

    def evaluate(self):
        if self.individual_img is not None:
            img_real = cv2.imread('./128_256/without_mask_4_situation_by_csv/test_quantitative_as_gt/{}'.format(self.individual_img))
            img_fake = cv2.imread(self.path_to_project + '/fake_B/{}'.format(self.individual_img))

            img_mask_percentage = cv2.imread('./binary_mask_with_percentage/{}'.format(self.individual_img), cv2.IMREAD_GRAYSCALE)
            img_mask_percentage_size = img_mask_percentage.size
            white_pixels = cv2.countNonZero(img_mask_percentage)
            white_pixels_ratio = (white_pixels / img_mask_percentage_size) * 100

            score_mse = compare_mse(img_real, img_fake)
            score_psnr = compare_psnr(img_real, img_fake)
            score_ssim = compare_ssim(img_real, img_fake, multichannel=True)

            print('binary_mask_percentage: {} %'.format(white_pixels_ratio))
            print('mse: {}\npsnr: {}\nssim: {}'.format(score_mse, score_psnr, score_ssim))

        else:
            score_psnr_list = []
            score_ssim_list = []
            percentage_0_10_list_psnr = []
            percentage_0_10_list_ssim = []
            percentage_10_15_list_psnr = []
            percentage_10_15_list_ssim = []
            percentage_15_20_list_psnr = []
            percentage_15_20_list_ssim = []
            percentage_20_25_list_psnr = []
            percentage_20_25_list_ssim = []
            percentage_25_30_list_psnr = []
            percentage_25_30_list_ssim = []
            percentage_30_over_list_psnr = []
            percentage_30_over_list_ssim = []
            real_files = glob.glob('./128_256/without_mask_4_situation_by_csv/test_quantitative_as_gt/*.jpg')
            fake_files = glob.glob(self.path_to_project + '/fake_B/*.jpg')

            for file in natsorted(fake_files):
                get_file_name = os.path.basename(file)
                # print(get_file_name)
                img_real = cv2.imread('./128_256/without_mask_4_situation_by_csv/test_quantitative_as_gt/{}'.format(get_file_name))
                img_fake = cv2.imread(self.path_to_project + '/fake_B/{}'.format(get_file_name))

                img_mask_percentage = cv2.imread('./binary_mask_with_percentage/{}'.format(get_file_name), cv2.IMREAD_GRAYSCALE)
                img_mask_percentage_size = img_mask_percentage.size
                white_pixels = cv2.countNonZero(img_mask_percentage)
                white_pixels_ratio = (white_pixels / img_mask_percentage_size) * 100

                score_psnr = compare_psnr(img_real, img_fake)
                score_ssim = compare_ssim(img_real, img_fake, multichannel=True)

                data = [get_file_name, white_pixels_ratio, score_psnr, score_ssim]

                with open('{}/{}'.format(self.path_to_project, 'evaluation.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)

                score_psnr_list.append(score_psnr)
                score_ssim_list.append(score_ssim)

                if white_pixels_ratio < 15:
                    percentage_10_15_list_psnr.append(score_psnr)
                    percentage_10_15_list_ssim.append(score_ssim)
                elif white_pixels_ratio < 20:
                    percentage_15_20_list_psnr.append(score_psnr)
                    percentage_15_20_list_ssim.append(score_ssim)
                elif white_pixels_ratio < 25:
                    percentage_20_25_list_psnr.append(score_psnr)
                    percentage_20_25_list_ssim.append(score_ssim)
                elif white_pixels_ratio < 30:
                    percentage_25_30_list_psnr.append(score_psnr)
                    percentage_25_30_list_ssim.append(score_ssim)
                elif white_pixels_ratio >= 30:
                    percentage_30_over_list_psnr.append(score_psnr)
                    percentage_30_over_list_ssim.append(score_ssim)

                # if white_pixels_ratio < 10:
                #     percentage_0_10_list_psnr.append(score_psnr)
                #     percentage_0_10_list_ssim.append(score_ssim)
                # elif white_pixels_ratio < 15:
                #     percentage_10_15_list_psnr.append(score_psnr)
                #     percentage_10_15_list_ssim.append(score_ssim)
                # elif white_pixels_ratio < 20:
                #     percentage_15_20_list_psnr.append(score_psnr)
                #     percentage_15_20_list_ssim.append(score_ssim)
                # elif white_pixels_ratio < 25:
                #     percentage_20_25_list_psnr.append(score_psnr)
                #     percentage_20_25_list_ssim.append(score_ssim)
                # elif white_pixels_ratio < 30:
                #     percentage_25_30_list_psnr.append(score_psnr)
                #     percentage_25_30_list_ssim.append(score_ssim)
                # elif white_pixels_ratio >= 30:
                #     percentage_30_over_list_psnr.append(score_psnr)
                #     percentage_30_over_list_ssim.append(score_ssim)

            score_psnr_mean = mean(score_psnr_list)
            score_ssim_mean = mean(score_ssim_list)

            score_psnr_variance = variance(score_psnr_list)
            score_ssim_variance = variance(score_ssim_list)

            score_psnr_max = max(score_psnr_list)
            score_ssim_max = max(score_ssim_list)

            score_psnr_min = min(score_psnr_list)
            score_ssim_min = min(score_ssim_list)

            data_index = ['index', 'length', 'psnr_mean', 'ssim_mean', 'psnr_variance', 'ssim_variance',
                          'psnr_max', 'ssim_max', 'psnr_min', 'ssim_min']

            data_all = ['all', len(score_psnr_list),
                        score_psnr_mean, score_ssim_mean,
                        score_psnr_variance, score_ssim_variance,
                        score_psnr_max, score_ssim_max,
                        score_psnr_min, score_ssim_min]

            print('all')
            print('psnr_mean: {}\nssim_mean: {}\n'.format(score_psnr_mean, score_ssim_mean))
            print('psnr_variance: {}\nssim_variance: {}\n'.format(score_psnr_variance, score_ssim_variance))
            print('psnr_max: {}\nssim_max: {}\n'.format(score_psnr_max, score_ssim_max))
            print('psnr_min: {}\nssim_min: {}\n'.format(score_psnr_min, score_ssim_min))

            ################################################################################################

            # percentage_0_10_psnr_mean = mean(percentage_0_10_list_psnr)
            # percentage_0_10_ssim_mean = mean(percentage_0_10_list_ssim)
            #
            # percentage_0_10_psnr_variance = variance(percentage_0_10_list_psnr)
            # percentage_0_10_ssim_variance = variance(percentage_0_10_list_ssim)
            #
            # percentage_0_10_psnr_max = max(percentage_0_10_list_psnr)
            # percentage_0_10_ssim_max = max(percentage_0_10_list_ssim)
            #
            # percentage_0_10_psnr_min = min(percentage_0_10_list_psnr)
            # percentage_0_10_ssim_min = min(percentage_0_10_list_ssim)
            #
            # data_0_10 = ['0_10', len(percentage_0_10_list_psnr),
            #              percentage_0_10_psnr_mean, percentage_0_10_ssim_mean,
            #              percentage_0_10_psnr_variance, percentage_0_10_ssim_variance,
            #              percentage_0_10_psnr_max, percentage_0_10_ssim_max,
            #              percentage_0_10_psnr_min, percentage_0_10_ssim_min]

            ################################################################################################

            percentage_10_15_psnr_mean = mean(percentage_10_15_list_psnr)
            percentage_10_15_ssim_mean = mean(percentage_10_15_list_ssim)

            percentage_10_15_psnr_variance = variance(percentage_10_15_list_psnr)
            percentage_10_15_ssim_variance = variance(percentage_10_15_list_ssim)

            percentage_10_15_psnr_max = max(percentage_10_15_list_psnr)
            percentage_10_15_ssim_max = max(percentage_10_15_list_ssim)

            percentage_10_15_psnr_min = min(percentage_10_15_list_psnr)
            percentage_10_15_ssim_min = min(percentage_10_15_list_ssim)

            data_10_15 = ['10_15', len(percentage_10_15_list_psnr),
                          percentage_10_15_psnr_mean, percentage_10_15_ssim_mean,
                          percentage_10_15_psnr_variance, percentage_10_15_ssim_variance,
                          percentage_10_15_psnr_max, percentage_10_15_ssim_max,
                          percentage_10_15_psnr_min, percentage_10_15_ssim_min]

            ################################################################################################

            percentage_15_20_psnr_mean = mean(percentage_15_20_list_psnr)
            percentage_15_20_ssim_mean = mean(percentage_15_20_list_ssim)

            percentage_15_20_psnr_variance = variance(percentage_15_20_list_psnr)
            percentage_15_20_ssim_variance = variance(percentage_15_20_list_ssim)

            percentage_15_20_psnr_max = max(percentage_15_20_list_psnr)
            percentage_15_20_ssim_max = max(percentage_15_20_list_ssim)

            percentage_15_20_psnr_min = min(percentage_15_20_list_psnr)
            percentage_15_20_ssim_min = min(percentage_15_20_list_ssim)

            data_15_20 = ['15_20', len(percentage_15_20_list_psnr),
                          percentage_15_20_psnr_mean, percentage_15_20_ssim_mean,
                          percentage_15_20_psnr_variance, percentage_15_20_ssim_variance,
                          percentage_15_20_psnr_max, percentage_15_20_ssim_max,
                          percentage_15_20_psnr_min, percentage_15_20_ssim_min]

            ################################################################################################

            percentage_20_25_psnr_mean = mean(percentage_20_25_list_psnr)
            percentage_20_25_ssim_mean = mean(percentage_20_25_list_ssim)

            percentage_20_25_psnr_variance = variance(percentage_20_25_list_psnr)
            percentage_20_25_ssim_variance = variance(percentage_20_25_list_ssim)

            percentage_20_25_psnr_max = max(percentage_20_25_list_psnr)
            percentage_20_25_ssim_max = max(percentage_20_25_list_ssim)

            percentage_20_25_psnr_min = min(percentage_20_25_list_psnr)
            percentage_20_25_ssim_min = min(percentage_20_25_list_ssim)

            data_20_25 = ['20_25', len(percentage_20_25_list_psnr),
                          percentage_20_25_psnr_mean, percentage_20_25_ssim_mean,
                          percentage_20_25_psnr_variance, percentage_20_25_ssim_variance,
                          percentage_20_25_psnr_max, percentage_20_25_ssim_max,
                          percentage_20_25_psnr_min, percentage_20_25_ssim_min]

            ################################################################################################

            percentage_25_30_psnr_mean = mean(percentage_25_30_list_psnr)
            percentage_25_30_ssim_mean = mean(percentage_25_30_list_ssim)

            percentage_25_30_psnr_variance = variance(percentage_25_30_list_psnr)
            percentage_25_30_ssim_variance = variance(percentage_25_30_list_ssim)

            percentage_25_30_psnr_max = max(percentage_25_30_list_psnr)
            percentage_25_30_ssim_max = max(percentage_25_30_list_ssim)

            percentage_25_30_psnr_min = min(percentage_25_30_list_psnr)
            percentage_25_30_ssim_min = min(percentage_25_30_list_ssim)

            data_25_30 = ['25_30', len(percentage_25_30_list_psnr),
                          percentage_25_30_psnr_mean, percentage_25_30_ssim_mean,
                          percentage_25_30_psnr_variance, percentage_25_30_ssim_variance,
                          percentage_25_30_psnr_max, percentage_25_30_ssim_max,
                          percentage_25_30_psnr_min, percentage_25_30_ssim_min]

            ################################################################################################

            percentage_30_over_psnr_mean = mean(percentage_30_over_list_psnr)
            percentage_30_over_ssim_mean = mean(percentage_30_over_list_ssim)

            percentage_30_over_psnr_variance = variance(percentage_30_over_list_psnr)
            percentage_30_over_ssim_variance = variance(percentage_30_over_list_ssim)

            percentage_30_over_psnr_max = max(percentage_30_over_list_psnr)
            percentage_30_over_ssim_max = max(percentage_30_over_list_ssim)

            percentage_30_over_psnr_min = min(percentage_30_over_list_psnr)
            percentage_30_over_ssim_min = min(percentage_30_over_list_ssim)

            data_30_over = ['30_over', len(percentage_30_over_list_psnr),
                            percentage_30_over_psnr_mean, percentage_30_over_ssim_mean,
                            percentage_30_over_psnr_variance, percentage_30_over_ssim_variance,
                            percentage_30_over_psnr_max, percentage_30_over_ssim_max,
                            percentage_30_over_psnr_min, percentage_30_over_ssim_min]

            ################################################################################################

            with open('{}/{}'.format(self.path_to_project, 'percentage_distribution.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data_index)
                writer.writerow(data_all)
                # writer.writerow(data_0_10)
                writer.writerow(data_10_15)
                writer.writerow(data_15_20)
                writer.writerow(data_20_25)
                writer.writerow(data_25_30)
                writer.writerow(data_30_over)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('arg1', help='name_of_log')  # for example, logs_conv3d_2d_cyclegan
    parser.add_argument('-img', '--arg2', help='evaluate_individual_img')

    args = parser.parse_args()

    RA = ResultArrangement(args.arg1, args.arg2)
    RA.arrange()
    RA.evaluate()

