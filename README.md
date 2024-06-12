BatikNitikClassifier: CNN-Based Image Classification for Batik Nitik Patterns with Confusion Matrix Evaluation

Data Collection
The data used in this program comes from the “Batik Nitik 960” dataset, which was published in a journal by Agus Eko M. et al. with the title “Batik Nitik 960 Dataset for Classification, Retrieval, and Generator” in 2023. This dataset consists of 960 images covering 60 categories of Nitik batik. However, for the implementation of this program, only 5 categories are used, namely “sekar_kemuning”, “sekar_liring”, “sekar_duren”, “sekar_gayam”, and “sekar_pacar”. Each category consists of 15 images for training, a total of 75 images, and 1 image per category for testing, a total of 5 images.

Feature determination
The features used to classify nitik batik pattern images are extracted from the image itself. In the context of Convolutional Neural Networks (CNN), these features are the patterns of pixels in the image identified through the convolution layer. CNNs automatically learn to determine important features during the training process, such as edges, texture, and shape.
Examples of features recognized by the convolution layer may include sharp corners in nitik batik, distinctive patterns, or other important elements that help in identifying the category of nitik batik.

Characterization retrieval
Characterization is performed using the convolution layers of the CNN model. Each convolution layer in the network identifies more complex and abstract features of the input image. Pooling layers are used to reduce the dimensionality of the data, retain important features, and reduce overfitting. This feature retrieval allows the model to understand the image hierarchically, from simple features to more complex features, which ultimately allows the model to distinguish between different categories of batik nitik.

Recognition model selection
The selected recognition model is a Convolutional Neural Network (CNN), which consists of multiple convolution and pooling layers, followed by a fully connected layer. The model architecture includes multiple convolution layers with ReLU activation, a pooling layer, a normalization layer, and a dense layer to produce the classification output.
as the following program code:

model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),  # Layer Conv2D baru
    layers.MaxPooling2D(),  # Layer MaxPooling2D baru
    layers.Flatten(),  # Layer Flatten
    layers.Dense(128, activation='relu'),  # Layer Dense dengan 128 neuron dan aktivasi relu
    layers.Dense(num_classes)  # Layer Dense output dengan jumlah neuron sesuai dengan num_classes
])

The CNN model is trained using the collected nitik batik image dataset. The dataset is divided into a training set and a validation set with a ratio of 80:20. The model is trained with Adam's optimization and uses the Sparse Categorical Crossentropy loss function. Training is done for 200 epochs to ensure the model can converge well.
The following is the program code:

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 200
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs)

Evaluation
Model evaluation is performed using validation sets and performance metrics such as accuracy, as well as confusion matrix visualization to measure the model's classification performance against each category of nitik batik patterns. The confusion matrix provides an overview of the correct and incorrect predictions for each class, allowing further analysis of classification errors.
The following is the program code:

cm = confusion_matrix(true_labels, predictions, labels=class_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.show()

Results
From the evaluation results of the classification model, it can be concluded that the CNN model that has been built has succeeded in classifying nitik batik images very well. This is reflected in the perfect precision, recall, and F1-score values (1.00) for each category of nitik batik evaluated.
Specifically, the conclusions from the evaluation results are as follows:
- Precision: The model has perfect precision (1.00) in classifying each category of nitik batik. Precision measures how accurate the model is in identifying a particular category of nitik batik from all those predicted.
- Recall: The model also has a perfect recall value (1.00) for each category of nitik batik. Recall measures how well the model can identify all the examples that actually belong to a particular category of batik nitik from all the actual examples.
- F1-score: F1-score, which is the harmonic mean of precision and recall, also reaches the maximum value (1.00) for each batik nitik category. F1-score measures the balance between precision and recall, which is an indicator of the overall performance of the model in performing classification.
