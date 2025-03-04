import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import os
import cv2
import numpy as np
dir_name = r"D:\dataset\faces"
y = []
X = []
target_names = []
h = w = 300
n_samples = 0
class_names = []
person_id = 0
if not os.path.exists(dir_name):
    print(f"Directory {dir_name} does not exist.")
else:
    print(f"Directory {dir_name} found. Processing...")
    
    for person_name in os.listdir(dir_name):
        dir_path = os.path.join(dir_name, person_name)
        print(f"Checking directory: {dir_path}")
        
        if not os.path.isdir(dir_path):  # Skip if it's not a directory
            print(f"{dir_path} is not a directory. Skipping.")
            continue
        
        class_names.append(person_name)
        
        for image_name in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image_name)
            print(f"Processing image: {image_path}")
            
            img = cv2.imread(image_path)
            if img is None:  # Skip if the image is not read correctly
                print(f"Could not read image {image_path}. Skipping.")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray, (h, w))
            v = resized_image.flatten()
            X.append(v)
            y.append(person_id)
            n_samples += 1
        
        person_id += 1 
    y = np.array(y)
    X = np.array(X)
    target_names = np.array(class_names)
    if X.size > 0:
        n_features = X.shape[1]
        n_classes = len(class_names)

        print(y.shape, X.shape, target_names.shape)
        print("Number of samples:", n_samples)
        print("Total dataset size:")
        print("n_samples: %d" % n_samples)
        print("n_features: %d" % n_features)
        print("n_classes: %d" % n_classes)
    else:
        print("No images found or processed. Check the dataset directory and ensure images are readable.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
n_components = 150
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(n_row * n_col, len(images))):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()

plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape, X_test_pca.shape)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pca, y_train)
X_train_lda = lda.transform(X_train_pca)
X_test_lda = lda.transform(X_test_pca)
print("LDA transformation done...")

clf = MLPClassifier(random_state=1, hidden_layer_sizes=(10, 10), max_iter=1000, verbose=True)
clf.fit(X_train_lda, y_train)
print("Model weights:")
model_info = [coef.shape for coef in clf.coefs_]
print(model_info)

y_pred = []
y_prob = []
for test_face in X_test_lda:
    prob = clf.predict_proba([test_face])[0]
    class_id = np.where(prob == np.max(prob))[0][0]
    y_pred.append(class_id)
    y_prob.append(np.max(prob))

y_pred = np.array(y_pred)
prediction_titles = []
true_positive = 0
for i in range(y_pred.shape[0]):
    true_name = class_names[y_test[i]]
    pred_name = class_names[y_pred[i]]
    result = 'pred: %s, pr: %s\ntrue: %s' % (pred_name, str(y_prob[i])[0:3], true_name)
    prediction_titles.append(result)
    if true_name == pred_name:
        true_positive += 1

print("Accuracy:", true_positive * 100 / y_pred.shape[0])

X_test_images = [x.reshape((h, w)) for x in X_test]

plot_gallery(X_test_images, prediction_titles, h, w)
plt.show()