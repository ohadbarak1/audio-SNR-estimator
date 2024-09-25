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

[![GNU GPLv3 License][license-shield]][license-url]
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
]
Most of these noises are stochastic, with some transient events such as car horns in the traffic data.

The project provides two workflows: 
1. Building of a noise augmented dataset where the speech data and noise data are summed with a known SNR. This SNR is saved as the label for the resulting audio frames.
2. Training and testing of a CNN that reads the augmented audio and utilizes the SNR labels generated in the previous step to build a model that predicts SNR on given audio clips.


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

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

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
[license-shield]: https://img.shields.io/github/license/ohadbarak1/audio-SNR-estimator.svg?style=for-the-badge
[license-url]: https://github.com/ohadbarak1/audio-SNR-estimator/blob/normalize_data_v1.0/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/ohadbarak


