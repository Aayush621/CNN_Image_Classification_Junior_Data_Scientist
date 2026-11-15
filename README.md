# Animal Image Classification Using CNN

A PyTorch CNN that classifies animal images as **cat**, **dog**, or **wild**.

[Open the notebook](CNN_Image_Classification_JuniorDataScientist.ipynb)

## Dataset

**Dataset source:** [Assignment-provided Google Drive link](https://drive.google.com/file/d/1PxuNRcCHmedctrmYi14pVRc02rGoUXHD/view?usp=sharing).

- **Training:** 14,630 images
- **Validation:** 1,500 images
- **Original image size:** 512 × 512 RGB

## Approach

1. Checked image samples, class counts, corrupt files, and exact duplicates.
2. Resized images to 256 × 256, converted them to tensors, and normalized pixel values.
3. Used three convolutional blocks to extract features and fully connected layers to classify images.
4. Trained with Adam and saved the model with the highest validation accuracy.
5. Evaluated accuracy, loss, precision, recall, F1-score, and the confusion matrix.

## Experiments

| Configuration | Best validation accuracy |
|---|---:|
| Initial 128px model, 10 epochs; seed uncontrolled | 97.20% |
| 128px, seed 42, 10 epochs | 97.87% |
| 256px, seed 42, 10 epochs | 98.47% |
| **256px, seed 42, 15 epochs** | **98.73%** |
| LR reduced to 0.0001 after epoch 10 | 98.67% |
| Adam with weight decay 0.0001 | 98.13% |
| Classifier dropout 0.2 | 98.13% |

The last three changes used a total 15-epoch budget and seed 42. None improved accuracy over the selected configuration. AdamW code was prepared, but no completed result was reported.

## Final Model

- **Architecture:** Three convolutional blocks with 32, 64, and 128 filters
- **Settings:** 256px, Adam, learning rate 0.001, batch size 32, 15 epochs, seed 42
- **Regularization:** No dropout, batch normalization, or weight decay
- **Recorded result:** 98.73% validation accuracy, loss 0.0725, and 19 errors out of 1,500 images

These are recorded training results; the clean notebook starts fresh. Validation was used for model selection, and no separate test set was evaluated.

## Run

Open the notebook in **Google Colab**, select a **GPU runtime**, and run the cells in order. The notebook includes best-model saving, feature maps, single-image prediction, and image-variation checks. Results for the variation checks have not yet been reported.

Files are saved locally in `/content/cnn_classification_final`. Download them before ending the runtime; Google Drive is not mounted.
