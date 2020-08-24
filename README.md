# Underwater-Acoustic-Target-Classification-Based-on-Dense-Convolutional-Neural-Network
In oceanic remote sensing operations, underwater acoustic target recognition is always a difficult and extremely important task of sonar systems, especially in the condition of complex sound wave propagation characteristics.  Expensively learning recognition model for big data analysis is typically an obstacle for most traditional machine learning (ML) algorithms, whereas convolutional neural network (CNN), a type of deep neural network, can automatically extract features for accurate classification.  In this study, we propose an approach using a dense CNN model for underwater target recognition.  The network architecture is designed to cleverly re-use all former feature maps to optimize classification rate under various impaired conditions while satisfying low computational cost. In addition, instead of using time-frequency spectrogram images, the proposed scheme allows directly utilizing original audio signal in time domain as the network input data.  Based on the experimental results evaluated on the real-world dataset of passive sonar, our classification model achieves the overall accuracy of 98.85% at 0 dB signal-to-noise ratio (SNR) and outperforms traditional ML techniques, as well as other state-of-the-art CNN models.
