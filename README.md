**SecureAuth Kivy - Biometric Authentication App**

![SecureAuth Kivy Logo](url_to_logo)

**Overview:**
SecureAuth Kivy is a Python-based biometric authentication app designed for projects requiring secure and efficient user verification. The app integrates facial recognition, voice recognition, and fingerprint identification for a comprehensive authentication experience.

**Biometric Methods:**
1. **Facial Recognition:**
   - Utilizes OpenCV for facial feature analysis.
   - Requires users to provide an image for facial comparison.
   - **Note: Download Caffe_prototxt and Caffe_model from the OpenCV Face Detect project and update their paths in `first.py`.**

2. **Voice Recognition:**
   - Implements a custom voice model for distinctive vocal pattern authentication.
   - Users create individual folders for voice samples within the `voice_samples` directory.
   - Update the path to the voice samples folder in `voice_auth_model.py`.

3. **Fingerprint Identification:**
   - Utilizes the Sokoto Coventry Fingerprint Dataset (SOCOFing) for fingerprint comparison.
   - Users need to download the dataset from [SOCOFing Dataset](https://www.kaggle.com/datasets/ruizgara/socofing).
   - Allows users to provide static fingerprint images for comparison.

**Getting Started:**
1. Clone the repository: `git clone https://github.com/your_username/SecureAuth-Kivy.git`
2. Navigate to the project directory: `cd SecureAuth-Kivy`

**Installation:**
1. Install dependencies: `pip install -r requirements.txt`
2. Download the SOCOFing Dataset from the provided link and place it in the `data` directory.
3. Download Caffe_prototxt and Caffe_model from the OpenCV Face Detect project and update their paths in `first.py`.

**Usage:**
1. Run the app: `python main.py`
2. Follow the on-screen instructions for biometric authentication.
3. Ensure to create facial images using OpenCV Face Detect project and encode them for comparison.
4. Create individual folders for each person in the `voice_samples` directory for voice authentication.
5. Update the path to the voice samples folder in `voice_auth_model.py`.

**Note:**
- This project relies on the SOCOFing Dataset for fingerprint comparison. Ensure to download and place the dataset in the `data` directory.
- For facial recognition, use OpenCV to capture and encode facial images for comparison.
- For voice recognition, create individual folders for each person in the `voice_samples` directory and update the path in `voice_auth_model.py`.

**Contributing:**
- Feel free to contribute by opening issues or submitting pull requests.

**License:**
This project is licensed under the [MIT License](LICENSE).

**Acknowledgments:**
- OpenCV - [https://github.com/opencv/opencv](https://github.com/opencv/opencv)
- SOCOFing Dataset - [https://www.kaggle.com/datasets/ruizgara/socofing](https://www.kaggle.com/datasets/ruizgara/socofing)
