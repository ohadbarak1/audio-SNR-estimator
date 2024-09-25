<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![GNU GPLv3 License]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ohadbarak1/audio-SNR-estimator/blob/normalize_data_v1.0/">
    <img src="images/logo.jpg" alt="Logo" width="160" height="160">
  </a>

<h3 align="center">Estimation of SNR in speech audio</h3>

  <p align="center">
    project_description
    <br />
    <a href="https://github.com/ohadbarak1/audio-SNR-estimator/blob/normalize_data_v1.0/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/ohadbarak1/audio-SNR-estimator/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/ohadbarak1/audio-SNR-estimator/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Multiple Automatic Speech Recognitioni (ASR) systems exist. Their ability to transcribe speech depends on multiple factors.
In this project, I attempt to address one of those factors: Signal-to-Noise Ratio (SNR).
Principally speaking, the lower the SNR of a speech audio signal, the less likely a given SNR would transcribe it correctly.
This is of course dependent on the type of noise, and the training applied to the ASR to handle that particular type of noise.

As ASR systems are being developed and optimized for edge devices, a question arises: should a given audio signal be transcribed by an ASR running locally on a given edge device, or should the audio be sent to processing in the cloud? 
An ASR model on the cloud is likely to be larger, and better equiped to transcribe audio with low levels of SNR.
However, this would come at a cost of higher latency. For some applications, latency could be a more important factor than accuracy, and for others the situation could be reversed. 

In essence, this project is trying to provide an estimation of SNR for a very simplified case of clean speech signal with additive environmental noise. The idea being that a low-complexity / low-power model like this could be run on an input audio stream on an edge device and provide the SNR. This SNR estimation can be used by an ASR system to determine whether to run a local ASR model (if the SNR is high), or to pass the audio to an instance of the ASR running on the cloud (if the SNR is low).

The speech data are taken from LibriSpeech: [https://www.openslr.org/12/].
The environmental noises are from the TAU Urban Acoustic Scenes 2022 development dataset: [https://zenodo.org/records/6337421
].
Most of these noises are stochastic, with some transient events such as car horns in the traffic data.

The project provides two workflows: 
1. Building of a noise-augmented dataset where the speech data and noise data are summed with a random, known SNR, and split into training, validation and testing sets. The SNR is saved as the label for the resulting audio frames in each set. Other standard audio augmentations are applied as well (pitch shift, time dilation, time shift, spectral shaping).
2. Training and testing of a CNN that reads the augmented audio and utilizes the SNR labels generated in the previous step to build a model that predicts SNR on given audio clips. The current model I implemented is a regression model, but one may envision a classification model where the SNR is binned into ranges.

This image shows three spectrograms of the word 'six' being spoken using different augmentations with varying SNRs:
 [![SNR augmentation example][SNR-augmentation-example]]


 I used Keras for the ML model development, and Librosa for feature generation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
<!--
## Getting Started
This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.
-->

### Installation
1. install Python 3.8 for your platform
2. Build a Python environment
  ```
  python -m venv myvenv
  ```
3. activate the environment
  ```
  source myenv/bin/activate
  ```
4. Clone the repo
  ```
  git clone https://github.com/ohadbarak1/audio-SNR-estimator.git
  ```
5. Install required pip packages
  ```
  pip install -r requirements.txt
  ```
6. Change git remote url to avoid accidental pushes to base project
  ```
  git remote set-url origin ohadbarak1/audio-SNR-estimator
  git remote -v # confirm the changes
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
The main Makefile defines the working directories.

1) mk/build_packages.mk has targets for several example workflows for generating augmented data packages and labels. Data packages are saved as numpy files.
2) mk/train_models.mk has targets for example model training workflows.
3) The par directory contains json files for package building and model training workflows.
4) Take a look at par/data_defaults.json, which defines source directories for the foreground speech data and the background noise data, and the output directories for the augmented data files. Adapt these to your dir structure. Parameters for the desired augmentation and filterbank construction are also in this file.
5) Take a look at par/ConvNet2D_A.json, which defines a network architecture and hyperparameters. The way I implemented it, the only requirement is that the final layer contain a single neuron (the normalized SNR value).
6) There's support only for Conv, MaxPool, AvgPool, Dropout, Flatten and Dense Keras layers.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the GNU GPLv3 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Ohad Barak - ohadbarak@gmail.com

Project Link: [https://github.com/ohadbarak1/audio-SNR-estimator/blob/normalize_data_v1.0](https://github.com/ohadbarak1/audio-SNR-estimator/blob/normalize_data_v1.0)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-url]: https://github.com/ohadbarak1/audio-SNR-estimator/blob/normalize_data_v1.0/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/ohadbarak
[SNR-augmentation-example]: images/audio-SNR-augmentation-example.jpg



