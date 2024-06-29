## 📝 Table of Contents

- [⤵️ Description](#description-)
- [🧩 Project Modules](#project-modules-)
- [⚙️ How to use?](#usage-)
- [🏆 Contributors](#contributors-)
## ⤵️ Description <a name = "description"></a>
* This project tackles the challenging task of classifying Arabic fonts in images. It achieves an impressive accuracy of **97.25%** in distinguishing between different font types like Scheherazade New, Marhey, Lemonada, IBM Plex Sans Arabic, even when the text color, size, and orientation vary within the images.
## 🧩 Project Modules <a name = "project-modules"></a>
### Preprocessing 
* After analyzing the images and identifying any noise or issues, the following preprocessing steps are applied:
    * Salt and pepper noise is removed using a median filter.
    * Images are converted to grayscale.
    * Rotating images only that need rotation.
        * This is done by identifying Hough lines in the image, extracting the longest line, and rotating the image based on the orientation of that line.
    * Finally, thresholding is applied to the images, resulting in nearly all images having a black background and white text.
### Feature Extraction
* extract relevant features from images to distinguish between different fonts.
* Used Local phase quantization.
### Model Training 
* model is trained on the extracted features from training data.
* Trying multiple models, achieving the best accuracy with the logistic regression model.
  <br/>
  ![image](https://github.com/Menna-Ahmed7/Neural-Project/assets/110634473/6519d940-71f7-4b56-b042-aff03b58c4e8)

### Performance Analysis
* testing the model on test (unseen) data, and calculating the model accuracy. 
## ⚙️ How to use? <a name = "usage"></a>
### 🧱 Prerequisites:
  * Jupyter Notebook installed
### 🏃 How to run?
  * Put images in the test folder
  * Run the last cell in predict_self.ipynb file
  * You can find the **predicted labels** in result.txt and the **execution time** taken by the model to predict the label in time.txt file.
## 🏆 Contributors <a name = "contributors"></a>
<table>
  <tr>
    <td align="center">
    <a href="https://github.com/Menna-Ahmed7" target="_black">
    <img src="https://avatars.githubusercontent.com/u/110634473?v=4" width="150px;" alt="https://github.com/Menna-Ahmed7"/>
    <br />
    <sub><b>Mennatallah Ahmed</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MostafaBinHani" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119853216?v=4" width="150px;" alt="https://github.com/MostafaBinHani"/>
    <br />
    <sub><b>Mostafa Hani</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MohammadAlomar8" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119791309?v=4" width="150px;" alt="https://github.com/MohammadAlomar8"/>
    <br />
    <sub><b>Mohammed Alomar</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/mou-code" target="_black">
    <img src="https://avatars.githubusercontent.com/u/123744354?v=4" width="150px;" alt="https://github.com/mou-code"/>
    <br />
    <sub><b>Moustafa Mohammed</b></sub></a>
    </td>
  </tr>
 </table>

