import os
import scipy.io as sio
import numpy as np
from PIL import Image
from PIL import ImageFilter as IF
import time
import math
import random as rand
import ipdb

from contrastAnalysis import contrastAnalysis
from maskImageCal import maskImageCal
from hazePatchCal import hazePatchCal
from imageResize import imageResize

if __name__ == "__main__":

    # Paths
    # Path to root folder
    root = "../../realphotos/"
    # root = "../../haze-level/"
    # Path to read image: Load Image as a m*n*3 matrix
    photo_path = "../../realphotos/photos/"
    # photo_path = "../../haze-level/images/"
    # Path to labels
    label_path = "../../realphotos/labels.mat"
    # label_path = "../../haze-level/labels.txt"
    # Path to mesures
    # measure_path = "C:\\Users\\lily\\Desktop\\HazePhoto_Dataset\\measures.mat"
    # measure = sio.loadmat(measure_path)
    # measure = measure['measure']

    # --------------------------------------------------------------------------------------------------------------------
    # Get index of testing samples with flag_test
    # for haze_level
    # label_list = []
    # with open(label_path, "r") as f:
    #     label_data = f.readlines()
    #     for line in label_data:
    #         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
    #         level = int(line[-2:])//5 - 1
    #         label_list.append(level)
    # label = np.array(label_list)
    # label = label[np.newaxis,:]

    # for realphotos
    label = sio.loadmat(label_path)
    label = label['label']

    number_of_photos = label.shape[1]  # label is of size [1,number_of_photos]

    # Compute statistics of classes
    index_of_measure_0 = np.where(label == 0)[1]
    index_of_measure_1 = np.where(label == 1)[1]
    index_of_measure_2 = np.where(label == 2)[1]
    index_of_measure_3 = np.where(label == 3)[1]
    index_of_measure_4 = np.where(label == 4)[1]
    number_of_each_measure = [len(index_of_measure_0), len(index_of_measure_1), len(index_of_measure_2),
                              len(index_of_measure_3), len(index_of_measure_4)]

    # Use Dataset of pytorch
    # Randomly generate index of testing samples evenly from each class
    Percent_of_testing_samples = 0.1
    index_0 = rand.sample(range(number_of_each_measure[0]),
                          math.ceil(number_of_each_measure[0] * Percent_of_testing_samples))
    index_1 = rand.sample(range(number_of_each_measure[1]),
                          math.ceil(number_of_each_measure[1] * Percent_of_testing_samples))
    index_2 = rand.sample(range(number_of_each_measure[2]),
                          math.ceil(number_of_each_measure[2] * Percent_of_testing_samples))
    index_3 = rand.sample(range(number_of_each_measure[3]),
                          math.ceil(number_of_each_measure[3] * Percent_of_testing_samples))
    index_4 = rand.sample(range(number_of_each_measure[4]),
                          math.ceil(number_of_each_measure[4] * Percent_of_testing_samples))

    index_all = measure = np.concatenate((index_of_measure_0[index_0], index_of_measure_1[index_1],
                                          index_of_measure_2[index_2], index_of_measure_3[index_3],
                                          index_of_measure_4[index_4]))

    # Obtain the flag of testing samples
    flag_test = np.zeros(number_of_photos)
    flag_test[index_all] = 1

    # store image as npy
    data_all_train = []
    data_all_test = []
    data_resized_image_all_train = []
    data_resized_image_all_test = []
    data_contrastMap_image_all_train = []
    data_contrastMap_image_all_test = []
    label_all_train = []
    label_all_test = []
    label_resized_train = []
    label_resized_test = []
    label_contrast_train = []
    label_contrast_test = []

    # --------------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    # # read image labels
    # # apply only to the synthetic dataset of 3024 haze images
    # path_to_label = "/home/ML/haze-level/labels.txt"
    # path_to_label = "C:\\Users\\lily\\Desktop\\haze-level\\labels.txt"
    # labels = labelProcessor(path_to_label)
    # --------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    # Process photos
    for x in range(0, 2):

        print("Start to process image %d." % (x + 1))

        # Set path to save internal results of each image
        result_path = root + "results/" + str(x + 1) + "/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # read images
        # jpgfile = Image.open(photo_path + str(x + 1) + ".jpg")
        pngfile = Image.open(photo_path + str(x + 1) + ".jpg")
        image = np.array(pngfile)
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]

        # calculate the running time of the whole process
        # start_time = time.perf_counter()
        start_time = time.perf_counter()

        # Compute haze regions
        # Contrast map calculation
        contrast_time = time.perf_counter()
        contrastMap_image, darkChannelMap_dark = contrastAnalysis(image, result_path, result_path)
        print("Running time for contrast time is %f sec." % (time.perf_counter() - contrast_time))

        # Compute denoising contrast map - smoothed with mode filter
        m, n = contrastMap_image.shape
        patch_size_radius = min(m, n) // (20 * 2)  # radius of filter
        img = Image.fromarray(contrastMap_image)
        img.save(result_path + 'contrastMapImage.png')
        darkChannelMap_dark_image = Image.fromarray(darkChannelMap_dark)
        darkChannelMap_dark_image.save(result_path + 'darkChannelMap_dark_image.png')
        DenoisedImage = img.filter(IF.ModeFilter(patch_size_radius))
        DenoisedImage.save(result_path + 'DenoisedImage.png')

        # obtain mask image and boundary of the smallest box covering all haze regions
        mask, bound = maskImageCal(DenoisedImage, m, n)

        # Obtain haze-region images (dark channel and RGB) from original photo using bound
        haze_region_dark = darkChannelMap_dark[bound[0]:bound[1], bound[2]:bound[3]]
        haze_region_R = R[bound[0]:bound[1], bound[2]:bound[3]]
        haze_region_G = G[bound[0]:bound[1], bound[2]:bound[3]]
        haze_region_B = B[bound[0]:bound[1], bound[2]:bound[3]]
        haze_region_img = Image.fromarray(haze_region_dark)
        haze_region_img.save(result_path + 'hazeRegionImage.png')

        # resize the images for training and testing data
        target_size = np.array([64, 64])  # The size of images (height, width) we want to obtain
        resized_image = imageResize(image, target_size, result_path)

        # ----------------------------------------------------------------------------------------------------------------------
        # save RGB images as feature data of original photos
        data_R = R.flatten()
        data_G = G.flatten()
        data_B = B.flatten()
        a = label[0, x]
        # data_all = np.array(list(np.array([label[0, x]])) + list(data_R) + list(data_G) + list(data_B))
        data_all = image

        if flag_test[x] == 1:
            # with open(root + "test_org_64.bin", 'ab') as f:
            #     f.write(data_all)
            data_all_test.append(data_all)
            label_all_test.append(label[0, x])
        else:
            # with open(root + "train_org_64.bin", 'ab') as f:
            #     f.write(data_all)
            data_all_train.append(data_all)
            label_all_train.append(label[0, x])

        # save resized RGB images as feature data of original photos
        resized_image_R = np.array(resized_image[1])
        resized_image_G = np.array(resized_image[2])
        resized_image_B = np.array(resized_image[3])

        data_resized_image_R = resized_image_R.flatten()
        data_resized_image_G = resized_image_R.flatten()
        data_resized_image_B = resized_image_R.flatten()

        # data_resized_image_all = np.array(list(np.array([label[0, x]])) + list(data_resized_image_R)
        #                                       + list(data_resized_image_G) + list(data_resized_image_B))
        data_resized_image_all = resized_image[0]

        if flag_test[x] == 1:
            # with open(root + "test_train_resized.bin", 'ab') as f:
            #     f.write(data_resized_image_all)
            data_resized_image_all_test.append(data_resized_image_all)
            label_resized_test.append(label[0, x])
        else:
            # with open(root + "train_org_resized.bin", 'ab') as f:
            #     f.write(data_resized_image_all)
            data_resized_image_all_train.append(data_resized_image_all)
            label_resized_train.append(label[0, x])

        # save contrast RGB images as feature data of original photos
        data_contrastMap_image = contrastMap_image.flatten()
        # data_contrastMap_image_all = np.array(list(np.array([label[0, x]])) + list(data_contrastMap_image))
        data_contrastMap_image_all = data_contrastMap_image

        if flag_test[x] == 1:
            # with open(root + "test_train_contrast.bin", 'ab') as f:
            #     f.write(data_contrastMap_image_all)
            data_contrastMap_image_all_test.append(data_contrastMap_image_all)
            label_contrast_test.append(label[0, x])
        else:
            # with open(root + "train_org_contrast.bin", 'ab') as f:
            #     f.write(data_contrastMap_image_all)
            data_contrastMap_image_all_train.append(data_contrastMap_image_all)
            label_contrast_train.append(label[0, x])

        # ----------------------------------------------------------------------------------------------------------------------
        # Obtain haze patches
        number_of_patch = 0  # The total number of patches
        patch_size = 64
        max_patch_per_image = 100
        hazePatchCal(patch_size, max_patch_per_image, m, n, bound, mask, number_of_patch, darkChannelMap_dark, R, G, B,
                     result_path)

        # --------------------------------------------------------------------------------------------------------------------
        # write to binary file
        '''
                        dark = patch_dark.flatten()
                        R = patch_R.flatten()
                        G = patch_G.flatten()
                        B = patch_B.flatten()

                        data1 = np.array(list(np.array([1])) + list(R) + list(G) + list(B))

                        data2 = np.array(list(np.array([1])) + list(R) + list(G) + list(B) + list(dark))

                        #data3 = np.array(list(np.array([1])) + list(dark))


                        if flag_test_data[x] == 1:
                            #flag_test += 1
                            with open("C:\\Users\\lily\\Desktop\\bi_test_patch_org_64.bin", 'ab') as f:
                                f.write(data1)

                            with open("C:\\Users\\lily\\Desktop\\bi_test_patch_all_64.bin", 'ab') as f:
                                f.write(data2)

                            #with open("/home/ML/cifar10/cifar-10-batches-bin/test_patch_dark_64.bin", 'ab') as f:
                                #f.write(data3)



                        else:
                            #flag_train += 1
                            with open("C:\\Users\\lily\\Desktop\\bi_train_patch_org_64.bin", 'ab') as f:
                                f.write(data1)

                            with open("C:\\Users\\lily\\Desktop\\bi_train_patch_all_64.bin", 'ab') as f:
                                f.write(data2)

                            #with open("/home/ML/cifar10/cifar-10-batches-bin/train_patch_dark_64.bin", 'ab') as f:
                                #f.write(data3)

        '''
        # --------------------------------------------------------------------------------------------------------------------

        # print the running time
        print("Running time for image %d is %f sec." % (x, time.perf_counter() - start_time))

        # print("Number of patches is %d." % (number_of_patch))

    # ---------------------------------------------------------------------------------------------------------------------
    # write to npy file
    np.save(root + "data/" + "test_org_64.npy", np.array(data_all_test))
    np.save(root + "data/" + "train_org_64.npy", np.array(data_all_train))
    np.save(root + "data/" + "test_org_resized.npy", np.array(data_resized_image_all_test))
    np.save(root + "data/" + "train_org_resized.npy", np.array(data_resized_image_all_train))
    np.save(root + "data/" + "test_org_contrast.npy", np.array(data_contrastMap_image_all_test))
    np.save(root + "data/" + "train_org_contrast.npy", np.array(data_contrastMap_image_all_train))

    np.save(root + "data/" + "label_all_test.npy", np.array(label_all_test))
    np.save(root + "data/" + "label_all_train.npy", np.array(label_all_train))
    np.save(root + "data/" + "label_resized_test.npy", np.array(label_resized_test))
    np.save(root + "data/" + "label_resized_train.npy", np.array(label_resized_train))
    np.save(root + "data/" + "label_contrast_test.npy", np.array(label_contrast_test))
    np.save(root + "data/" + "label_contrast_train.npy", np.array(label_contrast_train))
    # ---------------------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------
    # dataset generation
    '''
        # read resized image
        dark = np.array(dark)
        R = np.array(R)
        G = np.array(G)
        B = np.array(B)

        dark = dark.flatten()
        R = R.flatten()
        G = G.flatten()
        B = B.flatten()

        data1 = np.array(list(np.array([label])) + list(R) + list(G) + list(B))

        data2 = np.array(list(np.array([label])) + list(R) + list(G) + list(B)+ list(dark))

        data3 = np.array(list(np.array([label]))+ list(dark))

        # --------------------------------------------------------------------------------------------------------------------
        # write to binary file
        if flag_test_data[x] == 1:
            flag_test += 1
            with open("/home/ML/cifar10/cifar-10-batches-bin/test_segorg_128.bin", 'ab') as f:
                f.write(data1)

            with open("/home/ML/cifar10/cifar-10-batches-bin/test_all_128.bin", 'ab') as f:
                f.write(data2)

            with open("/home/ML/cifar10/cifar-10-batches-bin/test_our_128.bin", 'ab') as f:
                f.write(data3)



            print('Processing testing image: No.', flag_test)

        else:
            flag_train += 1
            with open("/home/ML/cifar10/cifar-10-batches-bin/train_segorg_128.bin", 'ab') as f:
                f.write(data1)

            with open("/home/ML/cifar10/cifar-10-batches-bin/train_all_128.bin", 'ab') as f:
                f.write(data2)

            with open("/home/ML/cifar10/cifar-10-batches-bin/train_our_128.bin", 'ab') as f:
                f.write(data3)


            print('Processing training image: No.', flag_train)
        # --------------------------------------------------------------------------------------------------------------------
    '''

# ------------------------------------------------------------------------------------------------------------------------
# following are workable functions but not helpful to our algorithm

'''




#---------Image edge finder------------
    root = '/home/lily/Desktop/test photos/web photos/result/7/'
    jpgfile = Image.open(root + "RawHazeImage.png")
    edgeImage = jpgfile.filter(IF.FIND_EDGES)
    edgeImage.save(root + 'edgeImage.png')

# -------------------------------------------------------------------





#---------Find contours---------------------------------------------------------
    #find contour
    image = np.array(smoothImage)
    Contour = measure.find_contours(image, 0.8)
    ContourImage = Image.fromarray(Contour)
    ContourImage.save(result_path + '/' + 'ContourImage.png')





    #cannot label contours with different values
    #structure = np.ones([3,3])
    #image = np.array(smoothImage)
    #labels, features = nd.measurements.label(image,structure)




# -------------------------------------------------------------------













#---------Image Smoother-last choice to remove noise------------
    m, n = np.array(jpgfile).shape
    patch_size = (m + n) / (2 * 20)
    patch_size = math.floor(patch_size / 2)
    smoothImage = jpgfile.filter(IF.ModeFilter(patch_size))
    smoothImage.save('smoothImageO.png')


    smoothImage = smoothImage.filter(IF.ModeFilter(patch_size*5))
  #  smoothImage = smoothImage.filter(IF.ModeFilter(patch_size*2))
  #  smoothImage = smoothImage.filter(IF.ModeFilter(patch_size*2))
 #   smoothImage = smoothImage.filter(IF.ModeFilter(patch_size*2))
    smoothImage.save('smoothImage.png')

# -------------------------------------------------------------------








#---------Image Enhancement-No use------------


    enhancer = IE.Sharpness(jpgfile)
    factor = 1000
    enhancer.enhance(factor).save('enhancer.png')

# -------------------------------------------------------------------






#---------shape removal-No use------------

mask = mp.diamond(1)

    contrastMap_refined = mp.binary_erosion(contrastMap_image, mask).astype(np.uint8)
    img = Image.fromarray(contrastMap_refined)
    img.save(result_path + '/' + 'contrastMap_refined.png')
# -------------------------------------------------------------------

   # See edges
    #root = '/home/lily/Desktop/test photos/web photos/result/4/'
    #jpgfile = Image.open(root + "RawHazeImage.png")
    #edgeImage = jpgfile.filter(IF.FIND_EDGES)
    #edgeImage.save(root + 'edgeImage.png')


'''
