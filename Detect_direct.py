import cv2
import pickle
import numpy as np
from sklearn.svm import SVC


def calculate_hog(image):
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)
    hog_features = hog_features.flatten()
    return hog_features


def save_model_with_pickle(model, filename):
    # Save the model using pickle
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)


def load_model_with_pickle(filename):
    # Load the model using pickle
    with open(filename, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    return loaded_model


def train_svm(images, labels):
    hog_features = [calculate_hog(image) for image in images]
    hog_features = np.array(hog_features)

    svm = SVC(kernel='linear')
    svm.fit(hog_features, labels)

    return svm


def test_svm(svm, test_image):
    test_feature = calculate_hog(test_image)
    prediction = svm.predict([test_feature])

    return prediction[0]


def main():
    # # 读取正样本图像
    # positive_images = [cv2.imread(r'D:\iPhone_Plug\result_pdb_positive\lianjieqi/0positive0.jpg', cv2.IMREAD_GRAYSCALE),
    #                    cv2.imread(r'D:\iPhone_Plug\result_pdb_positive\lianjieqi/0positive1.jpg', cv2.IMREAD_GRAYSCALE),
    #                    cv2.imread(r'D:\iPhone_Plug\result_pdb_positive\lianjieqi/1positive0.jpg', cv2.IMREAD_GRAYSCALE)]
    # negative_images = [cv2.imread(r'D:\iPhone_Plug\result_pdb_positive\lianjieqi/2positive0.jpg', cv2.IMREAD_GRAYSCALE),
    #                    cv2.imread(r'D:\iPhone_Plug\result_pdb_positive\lianjieqi/2positive1.jpg', cv2.IMREAD_GRAYSCALE),
    #                    cv2.imread(r'D:\iPhone_Plug\result_pdb_positive\lianjieqi/3positive0.jpg', cv2.IMREAD_GRAYSCALE)]
    #
    # # Assign labels: 1 for positive, 0 for negative
    # labels_positive = np.ones(len(positive_images))
    # labels_negative = np.zeros(len(negative_images))
    #
    # # Combine positive and negative examples
    # all_images = positive_images + negative_images
    # all_labels = np.concatenate([labels_positive, labels_negative])
    #
    # # Train SVM model
    # svm_model = train_svm(all_images, all_labels)
    # save_model_with_pickle(svm_model, 'svm_model.pkl')
    ######### Test SVM model
    svm_model = load_model_with_pickle('svm_model.pkl')
    test_image = cv2.imread(r'D:\iPhone_Plug\result_pdb_positive\lianjieqi/1positive1.jpg', cv2.IMREAD_GRAYSCALE)
    result = test_svm(svm_model, test_image)
    # 输出结果
    if result == 1:
        print("The object is detected.")
    else:
        print("The object is not detected.")


if __name__ == "__main__":
    main()
